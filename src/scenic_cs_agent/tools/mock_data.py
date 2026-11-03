"""假景区数据（Mock）。全部虚构，仅供练习。

真实系统里这些数据来自景区资源库 / 票务系统接口。
这里用一个 dict 代替，工具函数只依赖"按景区名查到一条记录"这个能力，
以后换成真接口时，只改工具函数内部，不影响 ToolRuntime 和 Agent。
"""

from __future__ import annotations

# key：景区名（工具参数 scenic_name 要精确匹配这里的 key）
# value：
#   open_hours  开放时间说明
#   closed_days 闭园日列表（空列表表示全年开放）
#   tickets     票种 → 价格（元）
SCENICS: dict[str, dict] = {
    "云栖山": {
        "open_hours": "08:00-17:30（16:30 停止入园）",
        "closed_days": ["周一（法定节假日除外）"],
        "tickets": {"成人票": 120, "学生票": 60, "儿童票(1.2-1.5m)": 60, "老人票(60+)": 0},
    },
    "镜湖湿地": {
        "open_hours": "07:30-18:00",
        "closed_days": [],
        "tickets": {"成人票": 80, "学生票": 40},
    },
}
