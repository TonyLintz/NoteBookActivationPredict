def timing():
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            time1 = time.time()
            ret = func(*args, **kwargs)
            time2 = time.time()
            print(
                "%s function complete in %.2f mins"
                % (func.__name__, (time2 - time1) / 60)
            )
            return ret

        return func_wrapper

    return decorator
