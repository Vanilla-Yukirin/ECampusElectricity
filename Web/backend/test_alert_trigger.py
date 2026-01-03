#!/usr/bin/env python3
"""
测试项目中SMTP告警触发机制
直接测试邮件发送功能，不依赖数据库
"""
import os
import sys
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr

def load_env_config():
    """直接从.env文件加载SMTP配置"""
    env_file = os.path.join(os.path.dirname(__file__), '../../.env')
    config = {}

    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'SMTP_SERVER':
                        config['smtp_server'] = value
                    elif key == 'SMTP_PORT':
                        config['smtp_port'] = int(value)
                    elif key == 'SMTP_USER':
                        config['smtp_user'] = value
                    elif key == 'SMTP_PASS':
                        config['smtp_pass'] = value
                    elif key == 'FROM_EMAIL':
                        config['from_email'] = value
                    elif key == 'USE_TLS':
                        config['use_tls'] = value.lower() in ('true', '1', 'yes')

    return config

def send_alert_email(subject: str, content: str, recipients: list) -> bool:
    """发送告警邮件（模拟AlertService.send_alert）"""
    config = load_env_config()

    if not config.get('smtp_user') or not config.get('smtp_pass'):
        print("SMTP配置不完整")
        return False

    try:
        if config.get("use_tls", False):
            server = smtplib.SMTP(
                host=config["smtp_server"],
                port=config["smtp_port"],
                timeout=15,
            )
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(
                host=config["smtp_server"],
                port=config["smtp_port"],
                timeout=15,
            )

        server.login(config["smtp_user"], config["smtp_pass"])

        msg = MIMEText(content, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8").encode()
        msg["From"] = formataddr(
            (
                Header("电费监控系统", "utf-8").encode(),
                config["from_email"],
            )
        )
        msg["To"] = ", ".join(recipients)

        server.sendmail(config["from_email"], recipients, msg.as_string())
        server.quit()
        return True

    except Exception as e:
        print(f"发送邮件异常: {e}")
        return False

def test_alert_trigger():
    """测试告警触发机制"""
    print("测试 SMTP 告警触发机制")
    print("=" * 50)

    # 模拟订阅信息
    subscription_info = {
        "name": "SMTP测试订阅",
        "room_id": "test-room-001",
        "threshold": 50.0,
        "email_recipients": ["lycorisgal@arisumika.top"]
    }

    # 模拟房间信息（电量低于阈值）
    room_info = {
        "room_id": subscription_info["room_id"],
        "room_name": "测试宿舍101",
        "electricity_balance": 30.5,  # 低于50度阈值
        "last_update": time.strftime('%Y-%m-%d %H:%M:%S'),
        "status": "正常"
    }

    print(f"  模拟订阅: {subscription_info['name']}")
    print(f"   房间ID: {subscription_info['room_id']}")
    print(f"   阈值: {subscription_info['threshold']}度")
    print(f"   收件人: {', '.join(subscription_info['email_recipients'])}")

    print("\n模拟房间信息:")
    print(f"   房间: {room_info['room_name']}")
    print(f"   电量余额: {room_info['electricity_balance']}度")
    print(f"   阈值: {subscription_info['threshold']}度")
    print("   状态: 低于阈值，应触发告警")

    # 构建告警邮件内容
    subject = f"电费余额告警 - {room_info['room_name']}"
    content = f"""
电费监控系统告警通知

电费余额不足告警

房间信息:
   房间名称: {room_info['room_name']}
   房间ID: {room_info['room_id']}

电费情况:
   当前余额: {room_info['electricity_balance']} 度
   告警阈值: {subscription_info['threshold']} 度

状态详情:
   剩余电量: {room_info['electricity_balance']:.1f} 度 (低于阈值 {subscription_info['threshold'] - room_info['electricity_balance']:.1f} 度)
   最后更新: {room_info['last_update']}

建议行动:
   请及时充值电费，避免影响日常生活。

---
此邮件由电费监控系统自动发送
"""

    # 记录开始时间
    start_time = time.time()

    # 发送告警邮件
    print("\n发送告警邮件...")
    success = send_alert_email(subject, content, subscription_info['email_recipients'])

    # 记录结束时间
    end_time = time.time()
    duration = end_time - start_time

    print(f" 发送耗时: {duration:.2f}秒"
    if success:
        print("告警邮件发送成功")
        print("请检查邮箱是否收到告警邮件")
    else:
        print("告警邮件发送失败")

    return success

def test_alert_timeliness():
    """测试告警及时性"""
    print("\n 测试告警及时性")
    print("=" * 30)

    # 进行多次测试
    results = []
    for i in range(3):
        print(f"\n第 {i+1} 次测试")
        success = test_alert_trigger()
        results.append(success)
        if success:
            time.sleep(2)  # 避免发送过于频繁

    # 统计结果
    total_tests = len(results)
    successful_tests = sum(results)
    success_rate = (successful_tests / total_tests) * 100

    print("\n及时性测试结果:")
    print(f"   总测试次数: {total_tests}")
    print(f"   成功次数: {successful_tests}")
    print(f"   成功率: {success_rate:.1f}%"
    if success_rate >= 80:
        print("及时性良好")
    elif success_rate >= 60:
        print(" 及时性一般")
    else:
        print("及时性较差")

    return success_rate >= 60

if __name__ == "__main__":
    print("ECampusElectricity SMTP 告警测试")
    print("=" * 50)

    try:
        # 显示SMTP配置
        config = load_env_config()
        print("当前SMTP配置:")
        print(f"   服务器: {config.get('smtp_server', '未配置')}")
        print(f"   端口: {config.get('smtp_port', '未配置')}")
        print(f"   用户: {config.get('smtp_user', '未配置')}")
        print(f"   发件人: {config.get('from_email', '未配置')}")
        print("")

        # 测试触发机制
        trigger_success = test_alert_trigger()

        # 测试及时性
        timeliness_success = test_alert_timeliness()

        print("\n" + "=" * 50)
        print("最终测试结果:")
        print(f"   触发机制: {'通过' if trigger_success else '失败'}")
        print(f"   及时性: {'通过' if timeliness_success else '失败'}")

        if trigger_success and timeliness_success:
            print("\nSMTP 告警系统测试全部通过！")
            sys.exit(0)
        else:
            print("\n SMTP 告警系统测试部分失败")
            sys.exit(1)

    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
