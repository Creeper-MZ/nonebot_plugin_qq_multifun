import asyncio
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent

from .services import ollama


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
    ollama.add_user(user)
    response = ollama.get_request(user, message)
    print(f"发送给: {user}\n{response}")
    with open('record.txt', 'a', encoding='utf8') as f:
        f.write(f'{user}:{message} AI:{response}\n')
    return response


# Message queue
message_queue: asyncio.Queue = asyncio.Queue()


async def process_message_queue():
    while True:
        task = await message_queue.get()
        await task
        message_queue.task_done()


async def add_task_to_queue(coro):
    await message_queue.put(coro)



def log_error(error: Exception, context: str):
    # Log errors here
    print(f"Error in {context}: {str(error)}")

