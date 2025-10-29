"""Command-line interface for whisper-flow."""

import os
import subprocess
import warnings
from pathlib import Path
from typing import Annotated

import typer
from typer import Option

from .app import WhisperFlow

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress ALSA warnings
os.environ["ALSA_SUPPRESS_WARNINGS"] = "1"
os.environ["ALSA_PCM_CARD"] = "0"
os.environ["ALSA_PCM_DEVICE"] = "0"

app = typer.Typer(
    name="whisper-flow",
    help="AI-powered voice-to-text with context-aware processing.",
    add_completion=False,
    no_args_is_help=True,
)

# Type aliases for common options
ConfigDirOption = Annotated[
    Path,
    Option("--config-dir", help="Custom configuration directory"),
]


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo("whisper-flow 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        Option("--version", callback=version_callback, help="Show version and exit"),
    ] = False,
) -> None:
    """WhisperFlow - AI-powered voice-to-text with context-aware processing."""


@app.command("init-config")
def init_config(config_dir: ConfigDirOption = None):
    """Initialize configuration files with defaults."""
    flow_app = WhisperFlow(config_dir)
    flow_app.config.ensure_config_files()
    typer.echo(f"✓ Configuration files initialized in {flow_app.config.config_dir}")
    typer.echo("\nNext steps:")
    typer.echo("1. Set your OPENAI_API_KEY environment variable")
    typer.echo("2. Run 'whisper-flow validate' to check your setup")
    typer.echo("3. Run 'whisper-flow daemon' to start the background service")
    typer.echo("4. Use the configured hotkeys for voice input")


@app.command()
def daemon(
    config_dir: ConfigDirOption = None,
    foreground: Annotated[
        bool,
        Option("--foreground", "-f", help="Run in foreground (don't daemonize)"),
    ] = False,
    _worker: Annotated[
        bool,
        Option("--_worker", help="Internal flag for background worker.", hidden=True),
    ] = False,
):
    """Start the WhisperFlow daemon with system tray and global hotkeys."""
    from .daemon import WhisperFlowDaemon

    if not _worker and not foreground:
        # This is the initial launch, so we start the background worker.
        WhisperFlowDaemon(config_dir).run(foreground=False, _worker=False)
        return

    # This is the actual worker process (or foreground mode).
    try:
        daemon_instance = WhisperFlowDaemon(config_dir)
        daemon_instance.run(foreground=foreground, _worker=_worker)
    except KeyboardInterrupt:
        if foreground:
            typer.echo("\nDaemon stopped by user")
    except ImportError as e:
        typer.echo(f"Error: Missing dependency for daemon mode: {e}", err=True)
        typer.echo("Install with: pip install 'whisper-flow[daemon]'", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error starting daemon: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def stop(
    config_dir: ConfigDirOption = None,
):
    """Stop the running WhisperFlow daemon."""
    from .daemon import stop_daemon

    try:
        stop_daemon()
    except Exception as e:
        typer.echo(f"Error stopping daemon: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def status(config_dir: ConfigDirOption = None):
    """Show system status and configuration."""
    flow_app = WhisperFlow(config_dir)

    typer.echo("WhisperFlow Status")
    typer.echo("==================")
    typer.echo(f"Mode: {flow_app.mode}")
    typer.echo(f"Config Directory: {flow_app.config.config_dir}")

    # OpenAI configuration
    api_key_status = "Yes" if flow_app.config.openai_api_key else "No"
    typer.echo(f"OpenAI Configured: {api_key_status}")
    typer.echo()

    # Model configuration
    typer.echo("Models:")
    typer.echo(f"  Transcription: {flow_app.config.transcription_model}")
    typer.echo(f"  Completion: {flow_app.config.completion_model}")
    typer.echo()

    # Daemon configuration
    typer.echo("Daemon Configuration:")
    typer.echo(f"  Daemon Enabled: {'Yes' if flow_app.config.daemon_enabled else 'No'}")
    typer.echo(f"  Auto-stop Silence: {flow_app.config.auto_stop_silence_duration}s")
    typer.echo()

    # Hotkeys
    typer.echo("Hotkeys:")
    typer.echo(f"  🎤 Transcribe: {flow_app.config.hotkey_transcribe}")
    typer.echo(f"  🔴 Auto-Transcribe: {flow_app.config.hotkey_auto_transcribe}")
    typer.echo(f"  🤖 Command: {flow_app.config.hotkey_command}")
    typer.echo()

    # Audio configuration
    typer.echo("Audio:")
    device_text = flow_app.config.mic_device_index or "Default"
    typer.echo(f"  Device Index: {device_text}")
    typer.echo(f"  Sample Rate: {flow_app.config.sample_rate} Hz")
    typer.echo(f"  VAD Mode: {flow_app.config.vad_mode}")
    typer.echo()

    # System dependencies
    typer.echo("System Dependencies:")
    deps = {
        "xdotool": ["xdotool", "--version"],
        "xclip": ["xclip", "-version"],
        "xsel": ["xsel", "--version"],
        "notify-send": ["notify-send", "--version"],
        "wmctrl": ["wmctrl", "-m"],
    }

    for name, cmd in deps.items():
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            typer.echo(f"  {name}: ✓")
        except (subprocess.CalledProcessError, FileNotFoundError):
            typer.echo(f"  {name}: ✗")
    typer.echo()

    # Services
    typer.echo("Services:")
    transcription_available = "Yes" if flow_app.config.openai_api_key else "No"
    completion_available = "Yes" if flow_app.config.openai_api_key else "No"
    typer.echo(f"  Transcription Available: {transcription_available}")
    typer.echo(f"  Completion Available: {completion_available}")
    typer.echo()

    # Usage instructions
    typer.echo("Getting Started:")
    if not flow_app.config.openai_api_key:
        typer.echo("  1. Set OPENAI_API_KEY environment variable")
        typer.echo("  2. Run 'whisper-flow validate' to verify setup")
    else:
        typer.echo("  1. Run 'whisper-flow daemon' to start background service")
        typer.echo("  2. Use hotkeys for voice input:")
        typer.echo(
            f"     • {flow_app.config.hotkey_transcribe}: Push-to-talk transcription",
        )
        typer.echo(
            f"     • {flow_app.config.hotkey_auto_transcribe}: Auto-stop transcription",
        )
        typer.echo(f"     • {flow_app.config.hotkey_command}: Command mode (with AI)")
        typer.echo("  3. Press Escape to cancel any recording")


@app.command()
def validate(config_dir: ConfigDirOption = None):
    """Validate configuration and dependencies."""
    flow_app = WhisperFlow(config_dir)

    try:
        # Run comprehensive validation
        validation_results = flow_app.run_comprehensive_validation()

        # Display results
        total_tests = sum(len(tests) for tests in validation_results.values())
        passed_tests = sum(
            len([t for t in tests if t["status"] == "pass"])
            for tests in validation_results.values()
        )

        typer.echo(f"\nValidation Results: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            typer.echo("✅ All validations passed! System is ready.")
        else:
            typer.echo("❌ Some validations failed. Check the issues above.")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"Validation error: {e}", err=True)
        raise typer.Exit(1)


def main_entry():
    """Main entry point for the whisper-flow command."""
    app()


def dictation_entry():
    """Entry point for legacy dictation command - redirects to daemon."""
    typer.echo("The dictation command has been replaced with the daemon.")
    typer.echo("Use: whisper-flow daemon")
    typer.echo("Then use hotkeys for voice input.")


if __name__ == "__main__":
    main_entry()
