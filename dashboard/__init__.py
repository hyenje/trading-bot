"""대시보드 진입점 헬퍼"""

__all__ = ["run_dashboard", "set_bot"]


def run_dashboard(*args, **kwargs):
    from dashboard.app import run_dashboard as _run_dashboard

    return _run_dashboard(*args, **kwargs)


def set_bot(*args, **kwargs):
    from dashboard.app import set_bot as _set_bot

    return _set_bot(*args, **kwargs)
