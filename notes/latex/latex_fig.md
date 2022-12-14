#### LaTex的图片

**单张图片**
```
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth,height=0.5\textwidth]{figure/sample.png}
    \caption{图片标题}
    \label{fig:sample}
\end{figure}
```

**多张并排**
```
% 导言区
\usepackage{caption} % 子图
\usepackage{subcaption} % 子图

% 正文区
\begin{figure}[H]
\centering  %图片全局居中
\subfigure[name1]{
\label{Fig.sub.1}
\includegraphics[width=0.45\textwidth]{DV_demand}}
\subfigure[name2]{
\label{Fig.sub.2}
\includegraphics[width=0.45\textwidth]{P+R_demand}}
\caption{Main name}
\label{Fig.main}
\end{figure}
```
