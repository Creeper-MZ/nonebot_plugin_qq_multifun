import asyncio
import threading

from nonebot import on_command, on_message, get_driver
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent, Message
from nonebot.rule import to_me
from .logger import log_with_timestamp
from .config import CONFIG
from .commands import register_commands
from .utils import run_loop, run_comfyui_queue_in_thread, run_novelai_queue_in_thread, resolve_user, \
    process_message_queue, get_user_compose,ollama_task_queue,run_ollama_queue_in_thread
from .utils import run_comfyui_queue,run_novelai_queue
main_handler = on_message(rule=to_me(), priority=98)


@main_handler.handle()
async def handle_message(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent):

    log_with_timestamp("开始分析用户")
    user_compose = (await get_user_compose(event.get_user_id()))
    message = str(event.get_message())
    task = {"userID": event.user_id,"prompt": message,"event": event, "bot": bot, "cmd_name": "dialog", "user_compose": user_compose}
    log_with_timestamp("用户分析完成")
    ollama_task_queue.put(task)
    log_with_timestamp("LLM任务进入队列")
    await main_handler.finish("纳西妲思考一下下~~~")





register_commands()
driver = get_driver()


@driver.on_startup
async def start_message_queue_processor():
    #message_loop = asyncio.new_event_loop()
    ollama_loop = asyncio.new_event_loop()
    novelai_loop = asyncio.new_event_loop()
    comfyui_loop = asyncio.new_event_loop()
    #threading.Thread(target=run_novelai_queue_in_thread).start()
    #threading.Thread(target=run_comfyui_queue_in_thread).start()
    threading.Thread(target=run_ollama_queue_in_thread).start()
    threading.Thread(target=run_loop,args=(novelai_loop,)).start()
    threading.Thread(target=run_loop,args=(comfyui_loop,)).start()
    #threading.Thread(target=run_loop, args=(message_loop,)).start()
    #asyncio.run_coroutine_threadsafe(process_message_queue(),ollama_loop)
    asyncio.run_coroutine_threadsafe(run_novelai_queue(),novelai_loop)
    asyncio.run_coroutine_threadsafe(run_comfyui_queue(),comfyui_loop)
    #asyncio.run_coroutine_threadsafe(process_message_queue(),message_loop)

@driver.on_shutdown
async def cleanup():
    pass
