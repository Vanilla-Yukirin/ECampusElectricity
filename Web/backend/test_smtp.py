#!/usr/bin/env python3
"""
SMTP 配置测试脚本
测试 SMTP 配置的有效性以及发送邮件功能
"""
import os
import sys
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

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

# 加载配置
settings = type('Settings', (), load_env_config())()

class SMTPTester:
    """SMTP 测试类"""

    def __init__(self):
        self.config = load_env_config()
        # 设置默认值
        self.config.setdefault('smtp_server', 'smtp.qq.com')
        self.config.setdefault('smtp_port', 465)
        self.config.setdefault('smtp_user', 'your-email@qq.com')
        self.config.setdefault('smtp_pass', 'your-email-authorization-code')
        self.config.setdefault('from_email', self.config['smtp_user'])
        self.config.setdefault('use_tls', False)

    def test_connection(self) -> bool:
        """测试 SMTP 连接"""
        logger.info("测试 SMTP 连接...")
        try:
            if self.config.get("use_tls", False):
                server = smtplib.SMTP(
                    host=self.config["smtp_server"],
                    port=self.config["smtp_port"],
                    timeout=10,
                )
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(
                    host=self.config["smtp_server"],
                    port=self.config["smtp_port"],
                    timeout=10,
                )

            # 测试连接
            server.ehlo()
            logger.info("SMTP 连接成功")
            server.quit()
            return True

        except Exception as e:
            logger.error(f"SMTP 连接失败: {e}")
            return False

    def test_authentication(self) -> bool:
        """测试 SMTP 认证"""
        logger.info("测试 SMTP 认证...")
        try:
            if self.config.get("use_tls", False):
                server = smtplib.SMTP(
                    host=self.config["smtp_server"],
                    port=self.config["smtp_port"],
                    timeout=10,
                )
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(
                    host=self.config["smtp_server"],
                    port=self.config["smtp_port"],
                    timeout=10,
                )

            server.login(self.config["smtp_user"], self.config["smtp_pass"])
            logger.info("SMTP 认证成功")
            server.quit()
            return True

        except Exception as e:
            logger.error(f"❌ SMTP 认证失败: {e}")
            return False

    def test_send_email(self, test_recipient: str = None) -> bool:
        """测试发送邮件"""
        if not test_recipient:
            test_recipient = self.config["smtp_user"]  # 发送给自己

        logger.info(f"测试发送邮件到: {test_recipient}")

        subject = "SMTP 配置测试"
        content = f"""
SMTP 配置测试邮件

此邮件用于测试 SMTP 配置是否正常工作。

发送时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
服务器: {self.config['smtp_server']}:{self.config['smtp_port']}
发件人: {self.config['from_email']}
收件人: {test_recipient}

如果您收到此邮件，说明 SMTP 配置正常。
"""

        try:
            if self.config.get("use_tls", False):
                server = smtplib.SMTP(
                    host=self.config["smtp_server"],
                    port=self.config["smtp_port"],
                    timeout=15,
                )
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(
                    host=self.config["smtp_server"],
                    port=self.config["smtp_port"],
                    timeout=15,
                )

            server.login(self.config["smtp_user"], self.config["smtp_pass"])

            msg = MIMEText(content, "plain", "utf-8")
            msg["Subject"] = Header(subject, "utf-8").encode()
            msg["From"] = formataddr(
                (
                    Header("电费监控系统测试", "utf-8").encode(),
                    self.config["from_email"],
                )
            )
            msg["To"] = test_recipient

            server.sendmail(self.config["from_email"], [test_recipient], msg.as_string())
            logger.info("邮件发送成功")
            server.quit()
            return True

        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")
            return False

    def run_all_tests(self, test_recipient: str = None):
        """运行所有测试"""
        logger.info("开始 SMTP 配置测试")
        logger.info("=" * 50)

        # 显示配置信息（隐藏密码）
        logger.info("当前 SMTP 配置:")
        logger.info(f"  服务器: {self.config['smtp_server']}:{self.config['smtp_port']}")
        logger.info(f"  用户名: {self.config['smtp_user']}")
        logger.info(f"  密码: {'*' * len(self.config['smtp_pass'])}")
        logger.info(f"  发件人: {self.config['from_email']}")
        logger.info(f"  TLS: {self.config['use_tls']}")
        logger.info("")

        results = []

        # 测试连接
        results.append(("连接测试", self.test_connection()))

        # 测试认证
        results.append(("认证测试", self.test_authentication()))

        # 测试发送邮件
        results.append(("邮件发送测试", self.test_send_email(test_recipient)))

        # 显示结果汇总
        logger.info("=" * 50)
        logger.info("测试结果汇总:")
        all_passed = True
        for test_name, passed in results:
            status = "通过" if passed else "❌ 失败"
            logger.info(f"  {test_name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            logger.info("\n所有测试通过！SMTP 配置正常。")
        else:
            logger.info("\n部分测试失败，请检查配置。")

        return all_passed


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="SMTP 配置测试工具")
    parser.add_argument("--recipient", "-r", help="测试邮件收件人地址（默认为发件人自己）")
    parser.add_argument("--connection-only", action="store_true", help="仅测试连接")
    parser.add_argument("--auth-only", action="store_true", help="仅测试连接和认证")

    args = parser.parse_args()

    tester = SMTPTester()

    if args.connection_only:
        success = tester.test_connection()
    elif args.auth_only:
        success = tester.test_connection() and tester.test_authentication()
    else:
        success = tester.run_all_tests(args.recipient)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
