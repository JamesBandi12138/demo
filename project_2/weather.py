import argparse
import sys
import requests


API_URL = "http://101.35.2.25/api/tianqi/tqybip.php"


def get_weather(api_id: str, dev_key: str):
    r = requests.get(API_URL, params={"id": api_id, "key": dev_key}, timeout=10)
    data = r.json()
    if data.get("code") != 200:
        raise ValueError(data.get("msg", "查询失败"))
    return data


def show_weather(data: dict):
    now = data.get("nowinfo", {})
    print("\n" + "=" * 40)
    print(f"{data.get('guo', '')} {data.get('sheng', '')} {data.get('shi', '')}")
    print("=" * 40)
    print(f"天气: {data.get('weather1')}")
    print(f"温度: {now.get('temperature')}°C (体感 {now.get('feelst')}°C)")
    print(f"湿度: {now.get('humidity')}%")
    print(f"风向: {now.get('windDirection')} ({now.get('windDirectionDegree')}°)")
    print(f"风速: {now.get('windSpeed')} km/h ({now.get('windScale')})")
    print(f"气压: {now.get('pressure')} hPa")
    print(f"时间: {now.get('uptime')}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="天气查询工具")
    parser.add_argument("-i", "--id", help="API ID")
    parser.add_argument("-k", "--key", help="开发者Key")
    args = parser.parse_args()

    if not args.id or not args.key:
        print("错误: 请提供API ID和Key")
        print("获取: www.apihz.cn")
        sys.exit(1)

    try:
        show_weather(get_weather(args.id, args.key))
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
