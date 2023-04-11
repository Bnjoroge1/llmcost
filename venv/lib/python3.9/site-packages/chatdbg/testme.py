import lldb

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f MyModule.helloWorld hello')

def helloWorld(debugger, command, result, internal_dict):
    print('Hello, world!')