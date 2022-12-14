#### LaTex的表格

```
\begin{table}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
\end{table}

% \\ 表格换行

% 添加样式
\begin{table}[htbp]
    1 & 2 & 3 \\
    4 & 5 & 6 \\
\end{table}

% h: 当前位置，将图形放置在正文文本给出该图像环境的地方。如果本页所剩的页面不够，这一参数不起作用。
% t: 顶部，将图形放置在页面顶部。
% b: 底部，将图形
% p: 浮动页，将图形放置在允许有浮动对象的页面上。


\begin{table}[htbp]
    \centering
    \hline
    1 & 2 & 3 \\
    4 & 5 & 6 \\
\end{table}

% \centering: 居中
% \hline: 横行
```

一个稍微复杂的table。
```
\begin{table}[htbp]
    \centering
    \caption{\label{tab:sensor}传感器介绍}
    
    \begin{tabularx}{\linewidth}{ p{.15\linewidth}<{\centering}  p{.25\linewidth}<{\centering}  p{.25\linewidth}<{\centering}  p{.25\linewidth}<{\centering}} 
        \hline
        传感器种类 & 优点 & 缺点 & 主要用途 \\ \hline
        \multirow{2}{*}{视觉传感器} & 范围广、数据多价格低廉 & 无法胡哦去环境的空间信息 & 识别路径及物体，增强视觉 \\
        \multirow{2}{*}{激光雷达} & 直接获取环境的空间信息 & 处理速度慢，价格昂贵 & 环境深度信息感知，三维重构 \\
        \multirow{2}{*}{毫米波雷达} & 能够近距离获得目标运动状态 & 分辨率低、易受干扰 & 获取目标距离，速度等 \\
        超声波雷达 & 处理速度快 & 方向性差 & 近距离目标检测 \\
        红外线 & 精度高 & 分辨率低，距离短 & 夜视、红外成像 \\ \hline
    \end{tabularx}
\end{table}
```

###### 表格的行距调整

```
\begingroup
\setlength{\tabcolsep}{10pt} % Default value: 6pt
\renewcommand{\arraystretch}{1.5} % Default value: 1
\endgroup
```