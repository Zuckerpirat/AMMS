import logging
from typing import Callable
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


def build_scheduler(
    morning_scan: Callable,
    close_scan: Callable,
    daily_summary: Callable,
) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=ET)

    scheduler.add_job(
        morning_scan,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=35, timezone=ET),
        id="morning_scan",
        name="Morning momentum scan",
        misfire_grace_time=300,
    )

    scheduler.add_job(
        close_scan,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=55, timezone=ET),
        id="close_scan",
        name="End-of-day exit check",
        misfire_grace_time=300,
    )

    scheduler.add_job(
        daily_summary,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=5, timezone=ET),
        id="daily_summary",
        name="Daily summary",
        misfire_grace_time=300,
    )

    return scheduler
