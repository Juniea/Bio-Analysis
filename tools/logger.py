import time

class SyncLogger:
    def __init__(self, level: str = "INFO") -> None:
        self.level_map = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "SILENCE": 4,
        }
        self.level = self.level_map[str(level).strip().upper()]
        self.last_time_stamp = None

    @property
    def _time_stamp(self) -> str:
        curr_time_stamp = time.strftime("[%m/%d/%y %H:%M:%S]", time.localtime())
        if self.last_time_stamp is None or self.last_time_stamp != curr_time_stamp:
            self.last_time_stamp = curr_time_stamp
            return curr_time_stamp
        else:
            return ' ' * len(curr_time_stamp)

    @staticmethod
    def _role(role: str) -> str:
        return f"[{role}]"

    @staticmethod
    def _level(level: str) -> str:
        return f"{level:9s}"

    def _do_print(self, level_digit: int) -> bool:
        return level_digit >= self.level

    @staticmethod
    def _red(text: str) -> str:
        return f"\033[91m{str(text)}\033[0m"

    @staticmethod
    def _green(text:str) -> str:
        return f"\033[92m{str(text)}\033[0m"

    @staticmethod
    def _yellow(text:str) -> str:
        return f"\033[93m{str(text)}\033[0m"

    @staticmethod
    def _blue(text:str) -> str:
        return f"\033[94m{str(text)}\033[0m"

    def set_level(self, level: str) -> None:
        self.level = self.level_map[level]

    def info(self, role: str, message: str) -> str:
        if self._do_print(1):
            text = f"{self._time_stamp} {self._blue(self._level('INFO'))}{self._role(role)}: {message}"
            print(text)
            return text
        else:
            return ''

    def debug(self, role: str, message: str) -> str:
        if self._do_print(0):
            text = f"{self._time_stamp} {self._green(self._level('DEBUG'))}{self._role(role)}: {message}"
            print(text)
            return text
        else:
            return ''

    def warning(self, role: str, message: str) -> str:
        if self._do_print(2):
            text = f"{self._time_stamp} {self._yellow(self._level('WARNING'))}{self._role(role)}: {message}"
            print(text)
            return text
        else:
            return ''

    def error(self, role: str, message: str) -> str:
        if self._do_print(3):
            text = f"{self._time_stamp} {self._red(self._level('ERROR'))}{self._role(role)}: {message}"
            print(text)
            raise RuntimeError(f"{self._role(role)}: {message}")
        else:
            raise RuntimeError(f"{self._role(role)}: {message}")