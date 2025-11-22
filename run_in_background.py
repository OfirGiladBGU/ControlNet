import subprocess
import platform


def launch_background_task():
    # log_file = "train_sd21.log"
    # script_to_run = "tutorial_train_sd21.py"

    # log_file = "train_sd21_custom_stippling.log"
    # script_to_run = "tutorial_train_sd21_custom_stippling.py"

    log_file = "model_predict.log"
    script_to_run = "model_predict.py"

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
    launch_background_task()
