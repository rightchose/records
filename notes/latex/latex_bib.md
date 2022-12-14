##### 浙江大学硕士毕业论文指南
1、参考的[repo](https://github.com/TheNetAdmin/zjuthesis)
2、安装`vscode`以及`latex-workshop`插件。
3、安装`texlive`，或者其他的发行版。
4、[配置](https://zhuanlan.zhihu.com/p/38178015)
5、参考的[issue](https://github.com/TheNetAdmin/zjuthesis/issues/11)
在先前的基础上配置下两个`recipe`。`xelatex`这个`recipe`可以只编译正文，`bibxe`会带上参考文献编译以及目录图表等信息。为了速度可以平时选择`xelatex`。同时，这里还提供另一个命令`latexmk -xelatex -outdir=out zjuthesis`，会创建`out`目录生成编译结果。
```
{
    "latex-workshop.latex.recipe.default":"lastUsed",
    "latex-workshop.latex.outDir":"out",
    "latex-workshop.latex.recipes":[
        {
            "name": "xelatex",
            "tools": [
                "xelatex"
            ]
        },
        {
            "name": "bibxe",
            "tools": [
              "xelatex",
              "biber",
              "xelatex",
              "xelatex"
            ]
          }
    ],
    "latex-workshop.latex.tools":[
        {
            "name": "xelatex", 
            "command": "xelatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-output-directory=%OUTDIR%",
                "%DOC%"
            ],
            "env": {}
        },
        {
            "name": "biber",
            "command": "biber",
            "args": [
              "%OUTDIR%/%DOCFILE%"
            ],
            "env": {}
          }
    ]
}
```
6、为了后续方便使用，建议章节进行管理，也方便debug。
就研究生而言，修改`body/graduate/content.tex`内容为：
```

% 绪论
\inputbody{chapter1}
% 基础
\inputbody{chapter2}
% 深度补全
\inputbody{chapter3}
% 3D目标检测
\inputbody{chapter4}
% 总结与展望
\inputbody{chapter5}
```
同时在同级目录下创建好对应的`tex`文件。
这样做的好处：方便章节管理，例如debug等，随着论文页数增多，编译耗时也在增加，此时可以在`contex.tex`注释掉已经写好的章节，加快编译速度。
