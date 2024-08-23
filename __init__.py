import asyncio
from nonebot import on_command, on_message, get_driver
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent, Message
from nonebot.rule import to_me

from .config import CONFIG
from .commands import register_commands
from .services import ollama, novel_ai, comfy_api
from .utils import resolve_user, process_message, process_message_queue, add_task_to_queue

main_handler = on_message(rule=to_me(), priority=98)


@main_handler.handle()
async def handle_message(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent):
    user = await resolve_user(bot, event)
    message = str(event.get_message())
    response = await process_message(user, message)
    await add_task_to_queue(bot.send(event, response))


register_commands()
driver = get_driver()


@driver.on_startup
async def start_message_queue_processor():
    asyncio.create_task(process_message_queue())


@driver.on_shutdown
async def cleanup():
    pass
