# tkj创建，用于测试功能
def FA(fn):
    print("FA:")
    def warp():
        return "<a>" + fn() + "<a>"

    return warp


def FB(fn):
    print("FB:")
    def warp():
        return "<b>" + fn() + "<b>"

    return warp


@FA  # 相当于makebold(test1),也就是把当前函数作为入参传过去
def test1():
    print("test1():")
    return "test1"


@FB
def test2():
    print("test2:")
    return "test2"


@FA
@FB
def test3():  # 函数和装饰器是倒着执行的，从下往上，从内而外，一层层执行
    print("test3:")
    return "test3"


print("print test1==================>")
# print(test1())
# print("print test2==================>")
# print(test2())
# print("print test3==================>")
# print(test3())