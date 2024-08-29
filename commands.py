from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, PrivateMessageEvent, Message, MessageSegment, Event
from nonebot.typing import T_State
from nonebot.params import CommandArg, ArgStr
import PIL
from .config import NOVELAI_BASE_ARTIST, NOVELAI_NEGATIVE_PROMPT
from .utils import get_user_compose, novelai_task_queue, comfyui_task_queue, ollama_task_queue


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
                    "event": event, "bot": bot, "cmd_name": cmd_name, "user_compose": user_compose}
            ollama_task_queue.put(task)
        elif cmd_name == "dcn":
            task = {"userID": event.user_id, "prompt": prompt, "event": event, "bot": bot, "cmd_name": cmd_name,
                    "user_compose": user_compose}
            novelai_task_queue.put(task)
            await cmd.finish(message=f"已加入NAI绘画队列，当前位置：{novelai_task_queue.qsize()}", at_sender=True)
            # image = await (await get_user_compose(event.get_user_id())).novel_ai.generate_image(prompt)
        elif cmd_name == "dcnp":
            task = {"userID": event.user_id, "prompt": prompt, "event": event, "bot": bot, "cmd_name": cmd_name,
                    "user_compose": user_compose}
            novelai_task_queue.put(task)
            await cmd.finish(message=f"已加入NAI绘画队列，当前位置：{novelai_task_queue.qsize()}", at_sender=True)
        elif cmd_name == "dcf":
            task = {"userID": event.user_id, "prompt": prompt, "event": event, "bot": bot, "cmd_name": cmd_name,
                    "user_compose": user_compose}
            ollama_task_queue.put(task)
        elif cmd_name == "dccf":
            task = {"userID": event.user_id, "prompt": prompt, "event": event, "bot": bot, "cmd_name": cmd_name,
                    "user_compose": user_compose}
            comfyui_task_queue.put(task)
            await cmd.finish(message=f"已加入ComfyUI绘画队列，当前位置：{comfyui_task_queue.qsize()}", at_sender=True)
            # image = await (await get_user_compose(event.get_user_id())).comfy_api.get_comfy_request(prompt)

    return handle_draw, execute_draw


def register_commands():
    draw_commands = {
        "d": on_command("d", block=True),
        "dc": on_command("dc", block=True),
        "dn": on_command("dn", block=True),
        "dcn": on_command("dcn", block=True),
        "dcnp": on_command("dcnp", block=True),
        "dcf": on_command("dcf", block=True),
        "dccf": on_command("dccf", block=True)
    }

    for cmd_name, cmd in draw_commands.items():
        handle_draw, execute_draw = create_draw_handler(cmd_name, cmd)
        cmd.handle()(handle_draw)
        cmd.got(cmd_name, prompt="你想让纳西妲画什么？")(execute_draw)
    clear_mem = on_command("clear_mem", aliases={"清除记忆", "充值"}, block=True)

    @clear_mem.handle()
    async def handle_city_pre(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent):
        user = event.get_user_id()
        (await get_user_compose(event.get_user_id())).ollama.clear_user_message(user)
        print(user + " 执行了记忆删除")
        await cmd.finish(message='已执行记忆删除', at_sender=True)

    goback = on_command("goback", aliases={"撤回", "撤回消息"}, block=True)

    @goback.handle()
    async def handle_city_pre(bot: Bot, event: GroupMessageEvent | PrivateMessageEvent):
        user = event.get_user_id()
        await (await get_user_compose(event.get_user_id())).ollama.remove_last_message(user)
        print(user + " 消息撤回")
        await cmd.finish(message='消息已经撤回', at_sender=True)

    help = on_command("help", aliases={"菜单", "帮助"}, block=True)

    @help.handle()
    async def _(arg: Message = CommandArg()):
        await help.finish(MessageSegment.image("E:\\GOCQ\\CyberWaifu\\lhcbot\\src\\plugins\\nonebot_plugin_qq_multifun\\menu_new.png"))

    get_cf_model = on_command("get_cf_mod", aliases={"获取模型", "获取当前模型"}, block=True)

    @get_cf_model.handle()
    async def handle_city(bot: Bot, event: Event):
        msg = ""
        for i in (await get_user_compose(event.get_user_id())).comfy_api.model_name:
            msg += "代号: " + i + " 模型: " + (await get_user_compose(event.get_user_id())).comfy_api.model_name[
                i] + '\n'
        await cmd.finish(message=msg)

    set_cf_model = on_command("set_cf_model", aliases={"模型设置"}, block=True)

    @set_cf_model.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state["set_cf_model"] = arg.extract_plain_text().strip()

    @set_cf_model.got("set_cf_model", prompt="请输入模型代号")
    async def _(bot: Bot, event: Event, target_text: str = ArgStr("set_cf_model")):
        (await get_user_compose(event.get_user_id())).comfy_api.set_model(name=target_text)
        print("CF绘画模型设置为: ", (await get_user_compose(event.get_user_id())).comfy_api.model_name[target_text])
        await set_cf_model.finish("模型设置成功")

    tts = on_command("tts", aliases={"语音", "说话", "say"}, block=True)

    @tts.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state["tts"] = arg.extract_plain_text().strip()

    @tts.got("tts", prompt="你想让纳西妲说什么？")
    async def _(bot: Bot, event: Event, target_text: str = ArgStr("tts")):
        user_compose = (await get_user_compose(event.get_user_id()))
        await tts.finish(
            MessageSegment.record(user_compose.ollama.to_base64(user_compose.ollama.tts_trans(target_text))))

    set_art = on_command("set_art", aliases={"画师", "画师串", "前置提示", "角色"}, block=True)

    @set_art.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state["set_art"] = arg.extract_plain_text().strip()

    @set_art.got("set_art", prompt="请输入画师串")
    async def _(bot: Bot, event: Event, target_text: str = ArgStr("set_art")):
        if target_text == "默认" or target_text == "default":
            (await get_user_compose(event.get_user_id())).novel_ai.set_artist_prompt(NOVELAI_BASE_ARTIST)
            with open("E:\\GOCQ\\CyberWaifu\\lhcbot\\src\\plugins\\nonebot_plugin_qq_multifun\\user\\artists\\" + event.get_user_id(), 'w') as f:
                f.write(NOVELAI_NEGATIVE_PROMPT)
                f.close()
            await set_art.finish("已恢复默认画师串")
        else:
            (await get_user_compose(event.get_user_id())).novel_ai.set_artist_prompt(target_text)
            with open("E:\\GOCQ\\CyberWaifu\\lhcbot\\src\\plugins\\nonebot_plugin_qq_multifun\\user\\artists\\" + event.get_user_id(), 'w') as f:
                f.write(target_text)
                f.close()
            print("画师串设置为: ", (await get_user_compose(event.get_user_id())).novel_ai.get_artist_prompt())
            await set_art.finish("画师串设置成功")


    set_neg = on_command("set_neg", aliases={"负面提示词", "设置负面提示词"}, block=True)

    @set_neg.handle()
    async def _(state: T_State, arg: Message = CommandArg()):
        if arg.extract_plain_text().strip():
            state["set_neg"] = arg.extract_plain_text().strip()

    @set_neg.got("set_neg", prompt="请输入negative prompt")
    async def _(bot: Bot, event: Event, target_text: str = ArgStr("set_neg")):
        if target_text == "默认" or target_text == "default":
            (await get_user_compose(event.get_user_id())).novel_ai.set_negative_prompt(NOVELAI_NEGATIVE_PROMPT)
            with open("E:\\GOCQ\\CyberWaifu\\lhcbot\\src\\plugins\\nonebot_plugin_qq_multifun\\user\\negative\\" + event.get_user_id(), 'w') as f:
                f.write(NOVELAI_NEGATIVE_PROMPT)
                f.close()
            await set_neg.finish("已恢复默认负面提示词")
        else:
            (await get_user_compose(event.get_user_id())).novel_ai.set_negative_prompt(target_text)
            with open("E:\\GOCQ\\CyberWaifu\\lhcbot\\src\\plugins\\nonebot_plugin_qq_multifun\\user\\negative\\" + event.get_user_id(), 'w') as f:
                f.write(target_text)
                f.close()
            print("负面提示词设置为: ", (await get_user_compose(event.get_user_id())).novel_ai.get_artist_prompt())
            await set_neg.finish("提示词设置成功")

    read_art = on_command("read_art", aliases={"读取设定", "读取"}, block=True)

    @read_art.handle()
    async def _(bot: Bot, event: Event):
        target_text = ""
        neg = ""
        with open("E:\\GOCQ\\CyberWaifu\\lhcbot\\src\\plugins\\nonebot_plugin_qq_multifun\\user\\artists\\" + event.get_user_id(), 'r') as f:
            for i in f.readlines():
                target_text += i
        with open("E:\\GOCQ\\CyberWaifu\\lhcbot\\src\\plugins\\nonebot_plugin_qq_multifun\\user\\negative\\" + event.get_user_id(), 'r') as f:
            for i in f.readlines():
                neg += i
        (await get_user_compose(event.get_user_id())).novel_ai.set_artist_prompt(target_text)
        (await get_user_compose(event.get_user_id())).novel_ai.set_negative_prompt(neg)
        print("画师串设置为: ", (await get_user_compose(event.get_user_id())).novel_ai.get_artist_prompt(),
              "负面提示设置为:", (await get_user_compose(event.get_user_id())).novel_ai.get_negative_prompt())
        await read_art.finish("数据读取成功")
