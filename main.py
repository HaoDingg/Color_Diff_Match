# -*- coding: utf-8 -*-
"""
Created on Thu May  8 23:05:52 2025

@author: Ding
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button


def rgb_to_lab(rgb):
    """将 RGB 转换为 LAB"""
    rgb = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    lab = rgb2lab(rgb / 255.0)
    return lab[0, 0]


def find_closest_color(image_b_lab, target_lab):
    """在 image_b_lab 中找到与 target_lab 色差最小的点"""
    delta_e = np.linalg.norm(image_b_lab - target_lab, axis=2)
    min_pos = np.unravel_index(np.argmin(delta_e), delta_e.shape)
    return min_pos, delta_e[min_pos]


class ColorMatcherGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("图像色差匹配工具")
        self.image_a = None
        self.image_b = None
        self.image_b_lab = None
        self.match_results = []
        
        # 图A上的点集合
        self.points_on_a = []
        
        # 图B相关变量
        self.fig_b = None
        self.ax_b = None
        self.fig_b_created = False
        self.points_on_b = []
        
        # 设置主窗口大小
        self.master.geometry("400x250")
        
        # 创建 Frame 来组织按钮
        frame = tk.Frame(master)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 使用更美观的按钮样式
        btn_style = {'font': ('微软雅黑', 10), 'width': 15, 'height': 2}
        
        self.btn_a = tk.Button(frame, text="选择图像 A", command=self.load_image_a, **btn_style)
        self.btn_b = tk.Button(frame, text="选择图像 B", command=self.load_image_b, **btn_style)
        self.btn_match = tk.Button(frame, text="寻找匹配", command=self.find_color_match, **btn_style, 
                             bg='#4CAF50', fg='white')
        self.btn_results = tk.Button(frame, text="显示结果", command=self.show_results, **btn_style)
        self.btn_reset = tk.Button(frame, text="重置视图", command=self.reset_view, **btn_style)
        
        # 使用网格布局
        self.btn_a.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        self.btn_b.grid(row=0, column=1, pady=10, padx=10, sticky="ew")
        self.btn_match.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        self.btn_results.grid(row=1, column=1, pady=10, padx=10, sticky="ew")
        self.btn_reset.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # 配置列的权重，使其可以随窗口大小调整
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def load_image_a(self):
        path = filedialog.askopenfilename(title="选择图像 A")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.image_a = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                messagebox.showinfo("成功", "图像 A 加载成功")
            else:
                messagebox.showerror("错误", "无法加载图像 A")

    def load_image_b(self):
        path = filedialog.askopenfilename(title="选择图像 B")
        if path:
            img = cv2.imread(path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_b = rgb
                self.image_b_lab = rgb2lab(rgb / 255.0)
                messagebox.showinfo("成功", "图像 B 加载成功")
            else:
                messagebox.showerror("错误", "无法加载图像 B")

    def find_color_match(self):
        if self.image_a is None or self.image_b is None:
            messagebox.showerror("错误", "请先加载图像 A 和 B")
            return
        
        # 清空之前的匹配结果
        self.match_results.clear()
        self.points_on_a.clear()
        
        # 创建图B的窗口（如果还没创建）
        if not self.fig_b_created:
            self.create_image_b_figure()
        
        # 创建图A的窗口
        fig_a, ax_a = plt.subplots(figsize=(8, 6))
        ax_a.imshow(self.image_a)
        ax_a.set_title("图 A：点击一个或多个点匹配颜色（右键退出）")
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        plt.tight_layout()
        
        def onclick(event):
            if event.button is MouseButton.LEFT and event.xdata and event.ydata:
                x, y = int(event.xdata), int(event.ydata)
                rgb = self.image_a[y, x]
                lab = rgb_to_lab(rgb)
                pos, dE = find_closest_color(self.image_b_lab, lab)
                
                # 添加到匹配结果
                match_idx = len(self.match_results) + 1
                self.match_results.append(((x, y), (pos[1], pos[0]), dE))
                
                # 在图A上标记点
                point_a = ax_a.plot(x, y, marker='o', color='red', markersize=8)[0]
                text_a = ax_a.text(x+5, y+5, str(match_idx), color='white', fontweight='bold', 
                              bbox=dict(facecolor='red', alpha=0.7))
                self.points_on_a.append((point_a, text_a))
                fig_a.canvas.draw_idle()
                
                # 在图B上添加匹配点
                self.add_match_point_on_b(pos, match_idx)
                
                print(f"[✓] 点 ({x},{y}) → 匹配点 ({pos[1]},{pos[0]})，ΔE={dE:.2f}")
            
            elif event.button is MouseButton.RIGHT:
                plt.close(fig_a)
        
        fig_a.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def create_image_b_figure(self):
        """创建图B的显示窗口，带有平移功能，自动适应窗口大小"""
        self.fig_b, self.ax_b = plt.subplots(figsize=(10, 8))
        self.ax_b.imshow(self.image_b)
        self.ax_b.set_title("图 B：匹配结果（可平移）")
        
        # 自动适应窗口
        plt.tight_layout()
        
        # 移除坐标轴刻度，让图像更大
        self.ax_b.set_xticks([])
        self.ax_b.set_yticks([])
        
        def zoom_with_scroll(event):
            """使用鼠标滚轮缩放"""
            base_scale = 1.1
            
            xlim = self.ax_b.get_xlim()
            ylim = self.ax_b.get_ylim()

            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None:
                return

            if event.button == 'up':
                # 放大
                scale_factor = 1/base_scale
            elif event.button == 'down':
                # 缩小
                scale_factor = base_scale
            else:
                return

            x_center = (xlim[0] + xlim[1])/2
            y_center = (ylim[0] + ylim[1])/2
            
            x_width = (xlim[1] - xlim[0]) * scale_factor
            y_width = (ylim[1] - ylim[0]) * scale_factor
            
            self.ax_b.set_xlim([x_center - x_width/2, x_center + x_width/2])
            self.ax_b.set_ylim([y_center - y_width/2, y_center + y_width/2])
            
            self.fig_b.canvas.draw_idle()

        self.fig_b.canvas.mpl_connect('scroll_event', zoom_with_scroll)
        

        self.dragging = False
        self.x0, self.y0 = None, None
        
        def on_press(event):
            if event.button == MouseButton.MIDDLE:
                self.dragging = True
                self.x0, self.y0 = event.xdata, event.ydata
                
        def on_release(event):
            self.dragging = False
            
        def on_motion(event):
            if self.dragging and event.xdata and event.ydata:
                dx = event.xdata - self.x0
                dy = event.ydata - self.y0
                
                xlim = self.ax_b.get_xlim()
                ylim = self.ax_b.get_ylim()
                
                self.ax_b.set_xlim(xlim[0] - dx, xlim[1] - dx)
                self.ax_b.set_ylim(ylim[0] - dy, ylim[1] - dy)
                
                self.fig_b.canvas.draw_idle()
        
        self.fig_b.canvas.mpl_connect('button_press_event', on_press)
        self.fig_b.canvas.mpl_connect('button_release_event', on_release)
        self.fig_b.canvas.mpl_connect('motion_notify_event', on_motion)
        
        self.fig_b_created = True
        plt.show(block=False)

    def add_match_point_on_b(self, match_pos, idx):
        """在图B上添加匹配点"""
        y, x = match_pos  
        point = self.ax_b.plot(x, y, marker='x', color='red', markersize=10, markeredgewidth=2)[0]
        text = self.ax_b.text(x+5, y+5, str(idx), color='white', fontweight='bold',
                         bbox=dict(facecolor='red', alpha=0.7))
        
        self.points_on_b.append((point, text))
        self.fig_b.canvas.draw_idle()

    def reset_view(self):
        """重置图B的视图"""
        if self.fig_b_created:
            self.ax_b.set_xlim(0, self.image_b.shape[1])
            self.ax_b.set_ylim(self.image_b.shape[0], 0)
            self.fig_b.canvas.draw_idle()
            
    def show_results(self):
        if not self.match_results:
            messagebox.showinfo("无结果", "没有记录任何匹配")
            return

        result_window = tk.Toplevel(self.master)
        result_window.title("匹配结果列表")
        result_window.geometry("500x400")


        frame = tk.Frame(result_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(frame, text="色差匹配结果", font=("微软雅黑", 14, "bold"))
        title_label.pack(pady=(0, 10))


        text = tk.Text(frame, width=60, height=20, font=("微软雅黑", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        

        scrollbar = tk.Scrollbar(text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text.yview)

        text.insert(tk.END, "序号\t图A点坐标\t\t图B点坐标\t\t色差(ΔE)\n")
        text.insert(tk.END, "="*60 + "\n")

        for i, (src, tgt, dE) in enumerate(self.match_results):
            text.insert(tk.END, f"{i+1}\t({src[0]}, {src[1]})\t\t({tgt[0]}, {tgt[1]})\t\t{dE:.2f}\n")

        text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorMatcherGUI(root)
    root.mainloop()
