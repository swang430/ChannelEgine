import json
import pprint # 导入pprint模块，用于“漂亮地”打印
from channel_model_38901.parameters.parameter_loader import Parameter_Loader

# --- 准备工作 ---
# !!! 重要: 请确保这个路径指向你项目中的实际JSON文件 !!!
PARAM_FILE_PATH = 'channel_model_38901/parameters/parameters2_tr38901_v19.0.0.json'

# --- 加载参数 ---
# 使用我们之前分析过的 ParameterLoader 来加载文件
# 这行代码执行后，params 对象里就包含了所有数据
print(f"正在从 {PARAM_FILE_PATH} 加载参数...")
params = ParameterLoader.load_parameters(PARAM_FILE_PATH)
print("加载完成！")
print("-" * 20)


# --- 现在，我们可以用不同方式查看 params 里的内容了 ---

# 1. 查看特定场景 ('UMa') 的参数
print("\n--- 查看 'UMa' 场景的参数 ---")
uma_params = params.get_scenario_params("UMa")
# 使用 pprint 来格式化输出，indent=4 表示缩进4个空格
pprint.pprint(uma_params, indent=4)


# 2. 查看特定CDL模型 ('CDL-A') 的参数
print("\n--- 查看 'CDL-A' 模型的参数 ---")
cdl_a_params = params.get_cdl_model_params("CDL-A")
pprint.pprint(cdl_a_params, indent=4)


# 3. 将所有参数以格式化的JSON字符串形式打印出来
# 这是查看全部内容最好的方法，因为它非常清晰
print("\n--- 查看所有参数 (格式化输出) ---")
# 使用 json.dumps 可以将Python字典转换回格式化的JSON字符串
all_params_formatted = json.dumps(params.data, indent=4)
print(all_params_formatted)

# 你甚至可以把它保存到一个新文件里慢慢看
# with open("params_output.txt", "w") as f:
#     f.write(all_params_formatted)
# print("\n所有参数也已保存到 params_output.txt 文件中。")