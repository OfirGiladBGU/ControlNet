import subprocess
import platform


def launch_training():
    log_file = "train_sd21.log"
    script_to_run = "tutorial_train_sd21.py"

    # Open log file in write mode
    with open(log_file, "w") as log:
        # Start the process in background
        process = subprocess.Popen(
            ["python", script_to_run],
            stdout=log,
            stderr=log,
            start_new_session=True  # Detach from parent session
        )

        pid = process.pid
        print(f"\n✅ Launched '{script_to_run}' in background.")
        print(f"📁 Output is being logged to: {log_file}")
        print(f"🆔 Process ID (PID): {pid}")

        system = platform.system()
        if system == "Windows":
            print(f"🛑 To stop it, run: taskkill /PID {pid} /F")
        else:
            print(f"🛑 To stop it, run: kill {pid}")


if __name__ == "__main__":
    launch_training()
