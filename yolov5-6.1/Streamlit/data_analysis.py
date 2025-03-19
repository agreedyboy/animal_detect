import os
import streamlit as st
import pandas as pd
import cv2 as cv
import numpy as np
import torch

st.set_page_config(
        page_title="智能动物识别系统",
        page_icon="🐾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# 自定义CSS样式
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        box-shadow: 5px 0 15px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #4CAF50;
    }
    .st-ae {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                             r"C:\yolov5-6.1\runs\train\exp19\weights\best.pt")
        model.eval()
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

def detect_and_annotate(image, model):
    result = model(image)

    detections = result.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1 , y1 , x2 , y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f'{row["name"]} {row["confidence"]:.2f}'

        # 绘制渐变边框
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 150, 0), 3)
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv.putText(image, label, (x1+5, y1+15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return image

def get_file_list(suffix, path):
    """
    获取当前目录所有指定后缀名的文件名列表、绝对路径列表
    :return: 文件名列表、绝对路径列表
    """
    input_template_all = []
    input_template_all_path = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.split('.')[1] == suffix:
                input_template_all.append(name)
                input_template_all_path.append(os.path.join(root, name))

    return input_template_all, input_template_all_path

 #在 Streamlit 中，@st.cache_data 是一个装饰器，用于缓存函数的输出结果。
#它的主要作用是提高应用的性能，避免重复计算。
#当被装饰的函数被调用时, Streamlit 会检查函数的输入参数是否与之前调用时相同。
#如果相同，则直接返回缓存的结果，而无需重新执行函数体中的代码。
    
@st.cache_data
def load_data(file):
    #提取数据
    #将上传的文件读取为文字流
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    img = cv.cvtColor(img , cv.COLOR_BGR2RGB)

    #将图片处理为（640，640）大小的，便于后续使用yolo模型对其检测
    resized_img = cv.resize(img , (640, 640), interpolation=cv.INTER_AREA)
    return resized_img


def main():

    # 标题区域
    st.markdown(
        """
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
            <h1 style="color:white;text-align:center;">🦁 智能动物识别系统</h1>
            <p style="color:white;text-align:center;">基于YOLOv5的野生动物识别与标注工具</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 侧边栏区域
    with st.sidebar:
        st.markdown("## 🎯 操作面板")
        st.markdown("---")
        file = st.file_uploader(
            "📤 上传图片 (支持JPG/PNG/JPEG)",
            type=["jpg", "png", "jpeg"],
            help="最大支持10MB的图片文件"
        )

        if file:
            st.success("✅ 文件上传成功！") 
            st.markdown(f"""
                **文件信息**
                - 文件名：`{file.name}`
                - 文件类型：`{file.type}`
                - 文件大小：`{file.size // 1024} KB`
            """)

        st.markdown("---")
        st.markdown("""
        **使用说明：**
        1. 上传包含动物的图片
        2. 系统自动进行检测识别
        3. 查看右侧的检测结果
        4. 支持多张图片连续检测
        """)



     # 主内容区域
    col1, col2 = st.columns(2, gap="large")
    
    if file:
        with st.spinner("🔍 正在分析图片..."):
            try:
                # 图片处理
                img = load_data(file)

                # 显示原始图片
                with col1:
                    st.markdown("### 📸 原始图片")
                    st.image(img, use_container_width=True)

                # 加载模型并检测
                model = load_model()
                if model:
                    with st.spinner("🤖 正在进行目标检测..."):
                        annotated_img = detect_and_annotate(img.copy(), model)
                    
                    # 显示检测结果
                    with col2:
                        st.markdown("### 🔍 检测结果")
                        st.image(annotated_img, use_container_width=True)
                        
                        # 显示统计信息
                        results = model(img)
                        detections = results.pandas().xyxy[0]
                        if not detections.empty:
                            st.success(f"🎉 检测到 {len(detections)} 个动物目标")
                            st.dataframe(detections[["name", "confidence"]]
                                        .rename(columns={"name": "类别", "confidence": "置信度"})       
                                        .style.format({"置信度": "{:.2%}"}))
                        else:
                            st.warning("⚠️ 未检测到任何动物目标")

            except Exception as e:
                st.error(f"❌ 处理过程中发生错误: {str(e)}")
    else:
        col1.info("ℹ️ 请从左侧上传图片开始分析")

        st.markdown("""
                    <div style="height:480px; background:#f0f0f0; display:flex; align-items:center; justify-content:center; border-radius:10px">
                        <span style="color:#666">等待图片上传</span>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    
    main()