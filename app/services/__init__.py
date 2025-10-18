from .task_manager import TASKS
from .email_notifier import EmailNotifier

_email_notifier = EmailNotifier()
_email_notifier.attach_to_task_manager(TASKS)
