# available fields: backend, frontend, netlist, spec, rtl, general

# models: 

# llama2_7b, llama2_13b, llama3_8b_instruct, llama3_1_70b_instruct, llama3_1_8b_instruct, llama3_2_11b_vision_instruct, llama3_2_90b_vision_instruct,
# mistrial_7b, 
# gpt35, gpt4o, gpt4, gpt4_turbo, gpt4v, minigpt4_vicuna7b, 
# gemini_1.0_pro, gemini_1.5_pro, gemini_1.5_flash, 
# claude_3_sonnet, claude_3_haiku, claude_3_opus, 
# reka_core, reka_flash, reka_edge, 
# instructblip_flan_t5_xl, instructblip_flan_t5_xxl, blip2_flan_t5_xl, blip2_flan_t5_xxl, 
# chatglm3_6b, 
# kosmos2_patch14_224, 
# qwen_2_5_7b_instruct, qwen_2_5_72b_instruct, qwen_2_7b_instruct, qwen_2_72b_instruct, qwen_2_0_5b_instruct, qwen_vl, qwen_vl_plus, qwen_vl_max, qwen_vl_chat, 
# bunny_1_0_3b, 
# fuyu_8b, 
# hpt_1_5_edge, 
# intern_chat_7b, intern_chat_20b, internlm_xcomposer_vl_7b, internlm_xcomposer2_vl_7b, internlm2_chat_7b, internlm2_chat_20b, internlm2_5_chat_7b, internlm2_5_chat_20b, 
# internvl2_8b, internvl2_40b, internvl_chat_v1_5, internvl_mini_chat_2b_v1_5, 
# minicpm_v, minicpm_v2, minicpm3_4b, minicpm1b_sft_bf16, minicpm2b_sft_bf16, minicpm_llama3_v_2_5, 
# yi_vl_6b, yi_vl_34b, yi_chat_6b, yi_chat_34b

# 2024.11.18
# python main.py --field rtl --model llama3_1_70b_instruct (Done)
# python main.py --field rtl --model llama3_8b_instruct (Done)
# python main.py --field rtl --model llama3_2_11b_vision_instruct (Done)
# python main.py --field spec --model llama3_2_90b_vision_instruct
# python main.py --field rtl --model intern_chat_20b (done)
# python main.py --field rtl --model internvl_mini_chat_2b_v1_5 (Done)
# python main.py --field rtl --model internvl2_40b (Done)
# python main.py --field rtl --model internlm_xcomposer_vl_7b (Done)
# python main.py --field spec --model internlm_xcomposer2_vl_7b
# python main.py --field rtl --model qwen_2_7b_instruct (Done)
# python main.py --field rtl --model qwen_vl_chat
# python main.py --field spec --model qwen_vl
# python main.py --field rtl --model gpt4o (done)
# python main.py --field netlist --model gpt4
# python main.py --field rtl --model gpt4_turbo
# python main.py --field rtl --model gpt4v (done)

# python main.py --field backend --model reka_core
# python main.py --field backend --model reka_flash
# python main.py --field backend --model reka_edge
# python main.py --field backend --model chatglm3_6b

python main.py --field backend --model minicpm3_4b
python main.py --field frontend --model minicpm3_4b
python main.py --field netlist --model minicpm3_4b
python main.py --field spec --model minicpm3_4b
python main.py --field rtl --model minicpm3_4b