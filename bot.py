import asyncio
import io
import os
import re
import sys
import time
import argparse
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
import dotenv

from src.core.pipeline import parse_match_video
from src.models.match_data import MatchRecord

dotenv.load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".avi", ".mkv")
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


async def process_video_and_reply(
    video_attachment: discord.Attachment,
    reply_target: discord.Message,
    status_target: Optional[discord.Message] = None,
    interaction: Optional[discord.Interaction] = None
):
    """Downloads a video attachment, processes it through the extraction pipeline replies match JSON.
    """
    start_time = time.time()
    temp_filename = f"match_{int(time.time())}_{video_attachment.filename}"
    temp_video_path = os.path.join(TEMP_DIR, temp_filename)

    progress_data = {"phase": "Downloading video...", "detail": "0%"}
    stop_progress_loop = False

    async def progress_updater(msg: discord.Message):
        last_text = ""
        while not stop_progress_loop:
            await asyncio.sleep(3.5)
            curr_text = f"⏳ **VirtualCM StatReader:** {progress_data['phase']}\n*{progress_data['detail']}*"
            if curr_text != last_text and not stop_progress_loop:
                try:
                    await msg.edit(content=curr_text)
                    last_text = curr_text
                except Exception:
                    pass

    if interaction:
        status_msg = await interaction.followup.send(
            f"⏳ **VirtualCM StatReader:** Downloading `{video_attachment.filename}`..."
        )
    else:
        status_msg = await reply_target.reply(
            f"⏳ **VirtualCM StatReader:** Downloading `{video_attachment.filename}`..."
        )
        try:
            await reply_target.add_reaction("⏳")
        except Exception:
            pass

    progress_task = asyncio.create_task(progress_updater(status_msg))

    try:
        progress_data["phase"] = "Downloading video..."
        progress_data["detail"] = f"{video_attachment.size / (1024 * 1024):.1f} MB"
        await video_attachment.save(temp_video_path)

        progress_data["phase"] = "Sampling frames & detecting UI stability..."
        progress_data["detail"] = "Filtering 3D background motion"

        def on_engine_progress(idx: int, total: int, timestamp_sec: float, msg: str):
            progress_data["phase"] = "Extracting player stats..."
            progress_data["detail"] = f"[{idx}/{total}] ({timestamp_sec:.1f}s) {msg}"

        match_record: MatchRecord = await asyncio.to_thread(
            parse_match_video,
            video_path=temp_video_path,
            sample_fps=10.0,
            progress_callback=on_engine_progress
        )

        elapsed = time.time() - start_time
        stop_progress_loop = True
        progress_task.cancel()

        json_bytes = io.BytesIO(
            match_record.model_dump_json(by_alias=True, exclude_none=True, indent=2).encode("utf-8")
        )
        clean_home = re.sub(r'[^a-zA-Z0-9]', '_', match_record.home_team.team_name).strip('_')
        clean_away = re.sub(r'[^a-zA-Z0-9]', '_', match_record.away_team.team_name).strip('_')
        json_filename = f"match_stats_{clean_home}_vs_{clean_away}.json"
        discord_file = discord.File(fp=json_bytes, filename=json_filename)

        if interaction:
            await interaction.followup.send(file=discord_file)
            try:
                await status_msg.delete()
            except Exception:
                pass
        else:
            await reply_target.reply(file=discord_file)
            try:
                await reply_target.remove_reaction("⏳", bot.user)
                await reply_target.add_reaction("✅")
            except Exception:
                pass
            try:
                await status_msg.delete()
            except Exception:
                pass

        print(f"[+] Successfully processed match: {match_record.home_team.team_name} vs {match_record.away_team.team_name}", flush=True)

    except Exception as exc:
        stop_progress_loop = True
        progress_task.cancel()
        err_msg = f"❌ **Error processing video:** {str(exc)}"
        print(f"[!] Extraction error: {exc}", file=sys.stderr, flush=True)

        if interaction:
            await interaction.followup.send(err_msg)
        else:
            await reply_target.reply(err_msg)
            try:
                await reply_target.remove_reaction("⏳", bot.user)
                await reply_target.add_reaction("❌")
            except Exception:
                pass

    finally:
        stop_progress_loop = True
        if not progress_task.done():
            progress_task.cancel()
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass


@bot.event
async def on_ready():
    """Bot startup event handler."""
    print("=" * 60, flush=True)
    print(f"Logged in as: {bot.user.name}#{bot.user.discriminator} (ID: {bot.user.id})", flush=True)
    print(f"Connected to {len(bot.guilds)} servers.", flush=True)
    print("Bot is ready", flush=True)
    print("Send a .mp4 or .mov video in chat to parse.", flush=True)
    print("=" * 60, flush=True)

    try:
        synced = await bot.tree.sync()
        print(f"[+] Synced {len(synced)} slash commands.", flush=True)
    except Exception as e:
        print(f"[!] Failed to sync slash commands: {e}", flush=True)


@bot.event
async def on_message(message: discord.Message):
    """Listens for messages containing match video attachments."""
    if message.author.bot:
        return

    video_att = None
    for att in message.attachments:
        fname = att.filename.lower()
        if fname.endswith(VIDEO_EXTENSIONS) or (att.content_type and att.content_type.startswith("video/")):
            video_att = att
            break

    if video_att is not None:
        print(f"[*] Detected video attachment from {message.author}: {video_att.filename} ({video_att.size / (1024*1024):.1f} MB)", flush=True)
        await process_video_and_reply(video_attachment=video_att, reply_target=message)

    await bot.process_commands(message)


@bot.tree.command(name="parse_match", description="Extract post match player data")
@app_commands.describe(video="Recorded match video file")
async def parse_match_command(interaction: discord.Interaction, video: discord.Attachment):
    fname = video.filename.lower()
    if not (fname.endswith(VIDEO_EXTENSIONS) or (video.content_type and video.content_type.startswith("video/"))):
        await interaction.response.send_message(
            "❌ Please upload a valid video file (`.mp4`, `.mov`, `.webm`).",
            ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)
    await process_video_and_reply(
        video_attachment=video,
        reply_target=interaction.channel,
        interaction=interaction
    )


def main():
    parser = argparse.ArgumentParser(description="Discord Bot")
    parser.add_argument("--token", help="Discord Bot Token (can also be set via DISCORD_BOT_TOKEN in .env)")
    args = parser.parse_args()

    token = args.token or os.getenv("DISCORD_BOT_TOKEN")

    if not token or token == "your_bot_token_here":
        print("\n" + "!" * 70, file=sys.stderr)
        print("ERROR: Discord Bot Token is missing!", file=sys.stderr)
        print("Please set DISCORD_BOT_TOKEN in your .env file or run with:", file=sys.stderr)
        print("  python bot.py --token <DISCORD_BOT_TOKEN>", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    bot.run(token)


if __name__ == "__main__":
    main()
