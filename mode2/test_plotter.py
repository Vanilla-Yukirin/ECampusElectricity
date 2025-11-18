"""
电费历史数据可视化测试脚本
用于测试 plotter.py 模块的绘图功能
"""

import sys
import json
from pathlib import Path

# 添加 src 目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from utils.plotter import Elect_plot


def main():
    """测试画图功能的主函数"""
    print("=" * 50)
    print("电费历史数据可视化测试")
    print("=" * 50)
    
    # 读取历史数据
    print("\n[1] 读取历史数据...")
    history_file = project_root / "data_files" / "his.json"
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {history_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误：{history_file} 不是有效的 JSON 文件")
        return
    
    if not history_data:
        print("❌ 错误：his.json 中没有数据！")
        return
    
    print(f"✓ 成功读取 {len(history_data)} 个寝室的历史记录")
    
    # 显示可用的寝室
    print("\n[2] 可用的寝室列表:")
    for idx, room_data in enumerate(history_data, 1):
        room_name = room_data.get("name", "未知")
        record_count = len(room_data.get("his", []))
        print(f"  {idx}. {room_name} ({record_count} 条记录)")
    
    # 选择第一个寝室进行测试
    test_room_data = history_data[0]
    test_room = test_room_data.get("name", "未知")
    room_history = test_room_data.get("his", [])
    
    if not room_history:
        print(f"\n❌ 错误：{test_room} 没有历史记录！")
        return
    
    print(f"\n[3] 选择测试寝室: {test_room} ({len(room_history)} 条记录)")
    
    # 初始化绘图器
    print("\n[4] 初始化绘图器...")
    assets_dir = project_root / "assets"
    plotter = Elect_plot(assets_dir)
    print("✓ 绘图器初始化成功")
    
    # 生成图表 - 使用正确的参数
    print("\n[5] 生成电费历史图表...")
    print(f"    房间名: {test_room}")
    print(f"    时间范围: 48小时")
    
    try:
        # plot_history 需要传入房间名和时间范围，它会自己读取 his.json
        result = plotter.plot_history(test_room, time_span=48)
        
        if result["code"] == 100:
            print(f"✓ 图表生成成功！")
            print(f"✓ 保存路径: {result['path']}")
        else:
            print(f"❌ 生成图表失败: [{result['code']}] {result['info']}")
            return
        
        # 显示数据摘要
        print("\n[6] 数据摘要:")
        values = [float(record.get("value", 0)) for record in room_history]
        print(f"  - 数据点数量: {len(room_history)}")
        print(f"  - 最高余额: {max(values):.2f} 度")
        print(f"  - 最低余额: {min(values):.2f} 度")
        print(f"  - 平均余额: {sum(values)/len(values):.2f} 度")
        
        # 测试消耗柱状图
        print("\n[7] 生成电费消耗柱状图...")
        result2 = plotter.plot_consumption_histogram(test_room, time_span=48)
        
        if result2["code"] == 100:
            print(f"✓ 柱状图生成成功！")
            print(f"✓ 保存路径: {result2['path']}")
        else:
            print(f"⚠ 柱状图生成失败: [{result2['code']}] {result2['info']}")
        
    except Exception as e:
        print(f"\n❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 50)
    print("测试完成！请查看生成的图片文件。")
    print("=" * 50)


if __name__ == "__main__":
    main()
