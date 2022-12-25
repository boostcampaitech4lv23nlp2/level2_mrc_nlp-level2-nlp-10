class Colors:
    INFO = "\033[94m[INFO] "
    END = "\033[0m"
    ERROR = "\033[91m[ERROR] "


def get_color(msg_type):
    if msg_type == "INFO":
        return Colors.INFO
    elif msg_type == "ERROR":
        return Colors.ERROR
    elif msg_type == "END":
        return Colors.END


def print_msg(msg, msg_type):
    color = get_color(msg_type)
    msg = "".join([color, msg, Colors.END])
    print(msg)
