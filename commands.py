from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent, Message, MessageSegment, Event
from nonebot.typing import T_State
from nonebot.params import CommandArg, ArgStr
from .utils import get_user_compose,novelai_task_queue,comfyui_task_queue,ollama_task_queue


def create_draw_handler(cmd_name, cmd):
    async def handle_draw(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent, state: T_State,
                          arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state[cmd_name] = arg.extract_plain_text().strip()

    async def execute_draw(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent,
                           target_text: str = ArgStr(cmd_name)):
        prompt = target_text
        image = ""
        await cmd.send("纳西妲开始画画啦！")
        user_compose = await get_user_compose(event.get_user_id())
        if cmd_name == "d":
            await cmd.send("纳西妲正在分析提示词哦")
            image = await (await get_user_compose(event.get_user_id())).ollama.get_sd_request_with_llama(prompt)
        elif cmd_name == "dc":
            image = await (await get_user_compose(event.get_user_id())).ollama.get_sd_request(prompt)
        elif cmd_name == "dn":
            task = {"userID": event.user_id,
                    "prompt": prompt,
                    "event": event, "bot": bot, "cmd_name": cmd_name,"user_compose": user_compose}
            ollama_task_queue.put(task)
        elif cmd_name == "dcn":
            task={"userID":event.user_id,"prompt":prompt,"event":event,"bot":bot,"cmd_name":cmd_name,"user_compose":user_compose}
            novelai_task_queue.put(task)
            await cmd.finish(message=f"已加入NAI绘画队列，当前位置：{novelai_task_queue.qsize()}",at_sender=True)
            #image = await (await get_user_compose(event.get_user_id())).novel_ai.generate_image(prompt)
        elif cmd_name == "dcf":
            task = {"userID": event.user_id, "prompt": prompt, "event": event, "bot": bot, "cmd_name": cmd_name,"user_compose":user_compose}
            ollama_task_queue.put(task)
        elif cmd_name == "dccf":
            task = {"userID": event.user_id, "prompt": prompt, "event": event, "bot": bot, "cmd_name": cmd_name,"user_compose": user_compose}
            comfyui_task_queue.put(task)
            await cmd.finish(message=f"已加入ComfyUI绘画队列，当前位置：{comfyui_task_queue.qsize()}", at_sender=True)
            #image = await (await get_user_compose(event.get_user_id())).comfy_api.get_comfy_request(prompt)

    return handle_draw, execute_draw


def register_commands():
    draw_commands = {
        "d": on_command("d", block=True),
        "dc": on_command("dc", block=True),
        "dn": on_command("dn", block=True),
        "dcn": on_command("dcn", block=True),
        "dcf": on_command("dcf", block=True),
        "dccf": on_command("dccf", block=True)
    }

    for cmd_name, cmd in draw_commands.items():
        handle_draw, execute_draw = create_draw_handler(cmd_name, cmd)
        cmd.handle()(handle_draw)
        cmd.got(cmd_name, prompt="你想让纳西妲画什么？")(execute_draw)
    clear_mem = on_command("clear_mem", block=True)

    @clear_mem.handle()
    async def handle_city_pre(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent):
        user = event.get_user_id()
        (await get_user_compose(event.get_user_id())).ollama.clear_user_message(user)
        print(user + " 执行了记忆删除")
        await cmd.finish(message='已执行记忆删除', at_sender=True)

    goback = on_command("goback", block=True)

    @goback.handle()
    async def handle_city_pre(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent):
        user = event.get_user_id()
        await (await get_user_compose(event.get_user_id())).ollama.remove_last_message(user)
        print(user + " 消息撤回")
        await cmd.finish(message='消息已经撤回', at_sender=True)

    help = on_command("help", block=True)

    @help.handle()
    async def _(arg: Message = CommandArg()):
        await help.finish(
            "直接聊天，调用ai大模型和纳西妲聊骚\n/clear_mem或者/清除记忆:清除对话记忆重新开始,如果遭遇审查，可以通过指令:/撤回 来撤回最后一轮消息，不必要清除记忆\n/tts 可以将文字转语音,听纳西妲说话\n/dn 通过ai翻译自然语言为prompt绘画调用nai3 api\n/dcn 自定义prompt直接绘画调用nai3 api\n/dcf 通过ai翻译自然语言为prompt绘画调用Comfy UI本地运行\n/dccf 自定义prompt直接绘画调用ComfyUI\nai翻译自然语言为提示词功能只能生成纳西妲角色\n/get_cf_mod获取可切换comfyui模型列表。\n因为存在问题切换模型功能暂不开放，仅留做调试")

    get_cf_model = on_command("get_cf_mod", block=True)

    @get_cf_model.handle()
    async def handle_city(bot: Bot, event: Event):
        msg = ""
        for i in (await get_user_compose(event.get_user_id())).comfy_api.model_name:
            msg += "代号: " + i + " 模型: " + (await get_user_compose(event.get_user_id())).comfy_api.model_name[i] + '\n'
        await cmd.finish(message=msg)

    set_cf_model = on_command("set_cf_model", block=True)

    @set_cf_model.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state["set_cf_model"] = arg.extract_plain_text().strip()
    @set_cf_model.got("set_cf_model", prompt="请输入模型代号")
    async def _(bot: Bot, event: Event, target_text: str = ArgStr("set_cf_model")):
        await (await get_user_compose(event.get_user_id())).comfy_api.set_model(name=target_text)
        print("CF绘画模型设置为: ", (await get_user_compose(event.get_user_id())).comfy_api.model_name[target_text])
        await set_cf_model.finish("模型设置成功")

    tts = on_command("tts", block=True)

    @tts.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state["tts"] = arg.extract_plain_text().strip()

    @tts.got("tts", prompt="你想让纳西妲说什么？")
    async def _(bot: Bot, event: Event, target_text: str = ArgStr("tts")):
        user_compose=(await get_user_compose(event.get_user_id()))
        await tts.finish(MessageSegment.record(user_compose.ollama.to_base64(user_compose.ollama.tts_trans(target_text))))