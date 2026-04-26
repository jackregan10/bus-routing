import csv
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def timestep_to_time_label(timestep, start_hour=8, minutes_per_step=1):
    total_minutes = start_hour * 60 + int(timestep) * minutes_per_step
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def make_human_readable_timetable(
    input_csv=None,
    output_txt=None,
    start_hour=8,
    minutes_per_step=1
):
    bus_events = defaultdict(list)

    if input_csv is None:
        input_csv = os.path.join(SCRIPT_DIR, "ppo_episode_chart.csv")
    if output_txt is None:
        output_txt = os.path.join(SCRIPT_DIR, "ppo_episode_human_readable.txt")

    with open(input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            bus_id = int(row["bus_id"])
            stop = int(row["stop"])
            event = row["event"]
            timestep = int(row["time"])

            clock_time = timestep_to_time_label(
                timestep,
                start_hour=start_hour,
                minutes_per_step=minutes_per_step
            )

            bus_events[bus_id].append({
                "timestep": timestep,
                "clock_time": clock_time,
                "stop": stop,
                "event": event
            })

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    with open(output_txt, "w") as f:
        f.write("Human-Readable PPO Evaluation Timetable\n")
        f.write("=" * 45 + "\n\n")

        f.write(f"Simulation start time: {start_hour:02d}:00\n")
        f.write(f"Minutes per timestep: {minutes_per_step}\n\n")

        for bus_id in sorted(bus_events):
            f.write(f"Bus {bus_id}\n")
            f.write("-" * 20 + "\n")

            events = sorted(bus_events[bus_id], key=lambda x: x["timestep"])

            for event in events:
                event_name = (
                    "Arrive at"
                    if event["event"] == "arrival"
                    else "Depart from"
                )

                f.write(
                    f"{event['clock_time']}  "
                    f"{event_name} Stop {event['stop']}\n"
                )

            f.write("\n")

    print(f"Saved human-readable timetable to {output_txt}")


make_human_readable_timetable(
    start_hour=8,
    minutes_per_step=1
)