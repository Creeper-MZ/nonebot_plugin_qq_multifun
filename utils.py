import asyncio
import queue

from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent, MessageSegment, Message

from .config import SYSTEM_PROMPTS, CONFIG
from .services import OllamaRam, NovelAIAPI, Comfy_API, userDir
from .logger import log_with_timestamp
import os

ollama_task_queue = queue.Queue()
comfyui_task_queue = queue.Queue()
novelai_task_queue = queue.Queue()
os.environ["PINECONE_API_KEY"] = CONFIG["PINECONE_API_KEY"]


def run_ollama_queue_in_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_message_queue())
async def process_message_queue():
    while True:
        task = ollama_task_queue.get()
        cmd_name=task["cmd_name"]
        if cmd_name == 'dialog':
            log_with_timestamp("LLM队列获取到任务")
            user_compose = task["user_compose"]
            log_with_timestamp("获取到用户")
            await user_compose.ollama.add_user(user_compose.userID)
            log_with_timestamp("LLM用户检索成功")
            response = await user_compose.ollama.get_claude_request(user_compose.userID,task["prompt"])
            #response = await user_compose.ollama.get_request(user_compose.userID, task["prompt"])
            log_with_timestamp("成功获得LLM响应")

            try:
                await asyncio.wait_for(task["bot"].send(event=task["event"], message=response, at_sender=True),timeout=1)
            except asyncio.TimeoutError:
                log_with_timestamp("LLM已执行超时强制跳过")
            log_with_timestamp("发送给: {" + str(task["userID"]) + "}\n"+response)
        elif cmd_name == 'dn' or cmd_name == 'dcf':
            log_with_timestamp("提示词分析队列获取到任务")
            user_compose = task["user_compose"]
            log_with_timestamp("获取到用户")
            if cmd_name == 'dn':
                draw_prompt=await user_compose.ollama.auto_prompt_nai_with_claude(task["prompt"])
                novelai_task_queue.put(
                    {"cmd_name": "dcn", "prompt": "[artist:ningen_mame],artist:ciloranko,[artist:mutou mato],[artist:painter-lhb],[artist:sho_(sho_lwlw)],[artist:tsubasa_tsubasa],year 2022,{{{nahida (genshin impact)}}},{{white hair,green_eyes}},{{{loli,child,petite,aged down,young age,slim body,slim leg,petite figure,little girl}}},,[[[[wlop]]]]," + draw_prompt + ",[[[artist:babu],[artist:kupuru (hirumamiyuu)],[artist:tianliang duohe fangdongye],[artist:agwing86]]]", "bot": task["bot"], "user_compose": user_compose,
                     "event": task["event"]})
            else:
                draw_prompt=await user_compose.ollama.auto_prompt_with_claude(task["prompt"])
                comfyui_task_queue.put(
                    {"cmd_name": "dccf", "prompt": "nahida_(genshin_impact),(loli,child,young age,slim_legs,petite,aged_down,slim_body,little_girl,underage)," + draw_prompt, "bot": task["bot"], "user_compose": user_compose,
                     "event": task["event"]})
            log_with_timestamp("提示词获取成功，转绘画队列")

            try:
                await asyncio.wait_for(task["bot"].send(event=task["event"], message=f"提示词获取成功，转入NAI绘画队列，当前位置：{novelai_task_queue.qsize()}", at_sender=True),
                                       timeout=1)
            except asyncio.TimeoutError:
                log_with_timestamp("DN已执行超时强制跳过")

def run_novelai_queue_in_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_novelai_queue())


async def run_novelai_queue():
    while True:
        task = novelai_task_queue.get()
        if task["cmd_name"] == 'dcn':
            log_with_timestamp("NAI队列获取到任务")
            user_compose = task["user_compose"]
            log_with_timestamp("获取到用户")
            image = await user_compose.novel_ai.generate_image(task["prompt"])
            log_with_timestamp("NAI图像生成完毕")
            try:
                # wait for a task to complete
                await asyncio.wait_for(task["bot"].send(event=task["event"], message=Message(
                    [MessageSegment.image(image), MessageSegment.text("纳西妲画完啦！")])), timeout=1)
            except asyncio.TimeoutError:
                log_with_timestamp("NAI已执行超时强制跳过")
            log_with_timestamp("NAI图像发送成功")
            log_with_timestamp("NAI任务执行完成，等待下一任务")


def run_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def run_comfyui_queue_in_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def run_comfyui_queue():
    while True:
        task = comfyui_task_queue.get()
        if task["cmd_name"] == 'dccf':
            log_with_timestamp("ComfyUI队列获取到任务")
            user_compose = task["user_compose"]
            log_with_timestamp("获取到用户")
            image = await user_compose.comfy_api.async_get_comfy_request(task["prompt"])
            log_with_timestamp("ComfyUI图像生成完毕")
            try:
                # wait for a task to complete
                await asyncio.wait_for(task["bot"].send(event=task["event"], message=Message(
                    [MessageSegment.image(image), MessageSegment.text("纳西妲画完啦！")]), at_sender=True), timeout=1)
            except asyncio.TimeoutError:
                log_with_timestamp("ComfyUI已执行超时强制跳过")
            log_with_timestamp("ComfyUI图像发送成功")
            log_with_timestamp("ComfyUI任务执行完成，等待下一任务")


class UserCompose:
    def __init__(self, userID):
        self.userID = userID
        self.ollama = OllamaRam(global_system_prompt=SYSTEM_PROMPTS["Nahida_Safe"])
        self.novel_ai = NovelAIAPI(CONFIG["NOVELAI_API_KEY"])
        self.comfy_api = Comfy_API()
        self.ollama.load_database()
        self.ollama.init_genai(userID)


async def get_user_compose(userID: str):
    user_compose: UserCompose
    if userID in userDir:
        user_compose = userDir[userID]
    else:
        userDir[userID] = UserCompose(userID)
        user_compose = userDir[userID]
    return user_compose


async def resolve_user(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        try:
            data = await bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id)
            return f"{data['user_name']}/{data['user_displayname']}"
        except Exception as e:
            print(f'获取群组成员失败: {event.user_id}-{e}')
            return str(event.user_id)
    else:
        try:
            friend_list = await bot.get_friend_list()
            for friend in friend_list:
                if str(friend['user_id']) == str(event.user_id):
                    return f"{friend['user_remark']}/{friend['user_name']}"
            return str(event.user_id)
        except Exception as e:
            print(f'获取好友信息失败: {event.user_id}-{e}')
            return str(event.user_id)


async def process_message(user: str, message: str) -> str:
    user_compose = (await get_user_compose(user)).ollama.add_user(user)
    response = user_compose.ollama.get_request(user, message)
    print(f"发送给: {user}\n{response}")
    with open('record.txt', 'a', encoding='utf8') as f:
        f.write(f'{user}:{message} AI:{response}\n')
    return response


# Message queue


def log_error(error: Exception, context: str):
    # Log errors here
    print(f"Error in {context}: {str(error)}")
