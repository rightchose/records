对于vscode而言，我们使用`LatexWorkshop`这一插件。参考其官方[[Home · James-Yu/LaTeX-Workshop Wiki · GitHub](https://github.com/James-Yu/LaTeX-Workshop/wiki)](https://github.com/James-Yu/LaTeX-Workshop/wiki)。

#### 预备

- 安装LaTex发行版，并配置系统路径。例如`Tex Live`。

- 使用`latexmk`作为默认的recipe去build LaTex项目。当然你可以去自己去编写你的LaTex recipe。

- 可供选择：安装ChkTex去关联LaTex项目。

#### 编译

##### 编译文档

**单个LaTex文件**通过command Palette中的`Build LaTex`，快捷键默认`ctrl`+`alt`+`b`编译。这个命令调用的recip定义在`latex-workshop.latex.recipe.default`。

**多文件 _TODO_**

依据个人需求你可以通过编写LaTex recipes去定义多个compiling toolchains用于构建 LaTex项目。然后通过`Build with recipe`选择合适的toolchain去构建项目。

下面介绍一些有帮助的settings。

- latex-workshop.latex.autoBuild.run：使用[the default (first) recipe](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile#latex-recipes)去自动构建LateX 项目。默认值`onFileChange`。

- latex-workshop.latex.recipes：building时运行的一系列tools 。JSON格式。

- latex-workshop.latex.tools：用于building的tools。JSON格式。

- latex-workshop.latex.magic.args：`Tex program`的arguments。字符串数组格式。

- latex-workshop.latex.magic.bib.args：`BIB program`的arguments。字符串数组格式。

- latex-workshop.latex.build.forceRecipeUsage：强制使用recipes。布尔类型。

##### 终止当前编译

Command Palette方式：调用`Kill LaTex compiler process`。

Tex badge方式：调用`Terminate current compilation`。

##### 自动构建LaTex

除了手动调用`Build LaTex Project`去编译文档，你也可以配置自动编译（当文档内容发生变化）。具体为配置`latex-workshop.latex.autoBuild.run`，其调用的recipe定义再`latex-workshop.latex.recipe.default`。

**latex-workshop.latex.autoBuild.run**

| 类型  | 默认值            | 可能的值                            |
| --- | -------------- | ------------------------------- |
| 字符串 | `onFileChange` | `never`，`onSave`，`onFileChange` |

其中，对于`onFileChange`，我们可以在构建前设置一个delay，具体可以通过`latex-workshop.latex.wath.delay`来配置。

**latex-workshop.latex.autoBuild.interval**

auto builds的间隔最小为1000ms。

**latex-workshop.latex.watch.files.ignore**

用于配置那些在auto build中忽略的文件。这个属性必须是一个模式数组。这些模式回合文件的绝对路径进行匹配。举个例子，如果想忽略`texmf`目录下的所有东西，可以将`**/texmf/**`添加进去。

**latex-workshop.latex.watch.usePolling**

使用polling去监测文件的变化。当Tex文件被网络硬盘或OneDrive替换，这个选项应该被打开。设置为true，可能会导致高CPU利用率。

**latex-workshop.latex.watch.interval**

polling的间隔，ms单位，reload vscode是的config中的改变生效。默认300ms。

**latex-workshop.latex.watch.delay**

开始build前的delay，单位ms，默认250ms。

##### LaTex recipes

A LaTex recipe 指得是commands数组。commands会被LaTex Workshop顺序执行。recipes被定义在`latex-workshop.latex.recipes`，默认的，LaTex Workshop包含两个基本的recipes，定义在`latex-workshop.latex.recipes`和`latex-workshop.latex.tools`：

- 第一个只依赖于`latexmk`命令。

- 第二个recipe顺序执行下面的commands，`pdflatex`$\rightarrow$`bibtex`$\rightarrow$`pdflatex`$\rightarrow$`pdflatex`。
  
  ```json
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk 🔃",
      "tools": [
        "latexmk"
      ]
    },
    {
      "name": "pdflatex ➞ bibtex ➞ pdflatex × 2",
      "tools": [
        "pdflatex",
        "bibtex",
        "pdflatex",
        "pdflatex"
      ]
    }
  ]
  ```

**LaTeX tools**

在Latex workshop配置中出现在recipes的tools中的tool，都被定义在`latex-workshop.latex.tools`。为了在reipe中使用tool，tool’s的`name`需要被包含在recipes的tools。默认的tool如下：

```json
"latex-workshop.latex.tools": [
  {
    "name": "latexmk",
    "command": "latexmk",
    "args": [
      "-synctex=1",
      "-interaction=nonstopmode",
      "-file-line-error",
      "-pdf",
      "-outdir=%OUTDIR%",
      "%DOC%"
    ],
    "env": {}
  },
  {
    "name": "pdflatex",
    "command": "pdflatex",
    "args": [
      "-synctex=1",
      "-interaction=nonstopmode",
      "-file-line-error",
      "%DOC%"
    ],
    "env": {}
  },
  {
    "name": "bibtex",
    "command": "bibtex",
    "args": [
      "%DOCFILE%"
    ],
    "env": {}
  }
]
```

可以看到的是，你可以创建多个recipes使用不同的tools。而对于每个tool都拥有一个`name`，一个`command`。除此外，tool还有`args`以及`env`这些field。其中，`env`是字典类型。例如，如果你想使用位于你`home`目录下的`texmf`子目录，可以按照下面这样写：

```json
  "env": {
      "TEXMFHOME": "%DIR%/texmf"
  }
```

这里你可以override环境变量。注意，在这里，只有`placeholders`，例如`%DIR%`可以发挥作用，或者其他变量`$PATH`，无法扩展。**_TODO 理解起来比较麻烦_**

**placeholders**

LaTex tools中的`args`和`env`两个参数可以包含使用`%`包裹的符号。LaTex Workshop注册了下面的一些placeholders。这些placeholder使用时会被运行时的一些信息替代。

| Placeholders                 | Replaced by                                              |
| ---------------------------- | -------------------------------------------------------- |
| `%DOC%`（`%DOC_32%`）          | without extension的root file的全部路径                         |
| `%DOCFILE%`                  | without extension的root file name                         |
| `%DOC_EXT%`（`%DOC_EXT_W32%`） | with extension的root file path                            |
| `%DOCFILE_EXT%`              | with extension的root file name                            |
| `%DIR%`（`%DIR_W32%`）         | root file directory                                      |
| `%TMPDIR%`                   | 用于保存辅助文件临时目录                                             |
| `%OUTDIR%`（`%OUTDIR_W32%`）   | `latex-workshop.latex.outDir`配置中的输出目录                    |
| `%WORKSPACE_FOLDER%`         | 当取工作目录路径                                                 |
| `%RELATIVE_DIR%`             | The root file directory relative to the workspace folder |
| `%RELATIVE_DOC%`             | file root file path relative to the workspace folder     |

因为大部分的LaTex编译器都接受root file name without extension，所以，`%DOC%`和`%DOCFILE%`不包含filename extension。然而，`texify`工具需要complete filename with its extension，此时就需要使用`%DOC_EXT%`。

**latex-workshop.latex.recipe.default**

该参数定义了`Build LaTex project`使用的recipe。除此之外，auto build也会使用到该参数。该配置中使用的recipes要和`latex-workshop.latex.recipes`中匹配。这里需要注意下两个特殊的值：

- `first`：使用定义在`latex-workshop.latex.recipes`的第一个。

- `lastUsed`：使用`LaTex Workshop:Build with recipe`刚用过的recipe。

**latex-workshop.latex.build.forceRecipeUsage**

Force the use of the recipe system even when a magic comment defines a TeX command.

*TODO 暂时不理解*

##### 多文件项目

对于一个LaTex项目，有着`\begin{document}`的文件视伪root file，会被作为该项目的入口。LaTex Workshop会智能地找到root file。

**the root file**

LaTex Workshop有着一套搜寻root file的方式：

- **Magic comment** ：`% !TEX root = relative/or/absolute/path/to/root/file.tex`。如果这个comments存在于the currently active editor，the referred file is set as root。你可以使用这个comand的`latex-workshp.addtextroot` 去帮助你插入这个magic comment。

- **Self check**：如果当前active editor 包含`\begin{document}`，就设置为root file。

- **Root directory check**： LaTex Workshop会迭代遍历workspace下的 the root folder中的所有`.tex`文件。第一个包含`\begin{docuent}`的视为root file。为了避免parsing  workspace中所有的 `.tex`文件，你可以缩小搜索范围。具体为使用`latex-workshop.latex.search.rootFiles.include`或`latex-workshop.latex.search.rootFiles.exclued`。

- TODO

**The dependencies**

在找到root file后，随后哦LaTex Workshop会去递归地搜索root file中使用`input`, `include`, `InputIfFileExists`, `subfile`, `import` and `subimport`所引用的文件。如果导入了一些external directories，你可以将这些extra directions配置到`latex-workshop.latex.texDirs`。

此外，对于一个有着和root file一样的basename的`.fls`文件而言，该文件会被用来计算一系列的依赖，例如，classes、packages、fonts、input `tex` files，listings、graphs等。当`latex-workshop.latex.autoBuild.run`设置为`onFileChange`，任何修改都会触发重新构建。你可以使用`latex-workshop.latex.watch.files.ignore`去避免一些文件被监测。默认会忽略其中的`.code.tex`或`.sty`后缀。

##### 相关的设置

**latex-workshop.latex.search.rootFiles.include**

root detection mechanism包含在的文件规则

**latex-workshop.latex.search.rootFile.exclude**

root detection mechanism 排除的文件规则

##### 捕获错误和警告

编译工具链产生的错误会在`Problems Pane`中显示。 

##### 清除生成文件

LaTeX编译过程中会生成一些辅助文件。对于这些文件可以从`Command Palette`中调用`Clean up auxiliary files`清除。`latex-workshop.clean`绑定再`ctrl`+`alt`+`c`。

##### 外部构建命令

尽管先前描述的recipe mechanism功能十分强大，但可能依旧无法满足你的需求，例如你可能需要执行一些个人脚本或makefile。对于这部分case，我们提供an external build command mechanism。

##### Magic comments

默认情况下magic comment是关闭的。你可以通过修改settings来支持该feature的使用。

**TeX program and options**

LaTeX Workshop支持`% !TEX program` magic comment 指定编译程序。然而，我们推荐使用recipe system 而不是magic comment。因为后者指对老的版本稳定。

**TODO 因为magic comments不太推荐，后面的一些不久看了**

#### Build a `.jnw` file

- [Building](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile#building-the-document)
- [Viewing and going from source to PDF back and forth](https://github.com/James-Yu/LaTeX-Workshop/wiki/View)
- [Catching errors and warnings](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile#catching-errors-and-warnings)
- [Navigating and selecting environments](https://github.com/James-Yu/LaTeX-Workshop/wiki/Environments#Navigating-and-selection)
- Navigating the document structure. The section names of LaTeX outline hierarchy are defined in [`latex-workshop.view.outline.sections`](https://github.com/James-Yu/LaTeX-Workshop/wiki/ExtraFeatures#latex-workshopviewoutlinesections). This property is an array of case-sensitive strings in the order of document structure hierarchy. For multiple tags in the same level, separate the tags with `|` as delimiters, e.g., `section|alternative`. It is also used by the folding mechanism.
- Miscellaneous actions
  - Open citation browser, see also [Intellisense for citations](https://github.com/James-Yu/LaTeX-Workshop/wiki/Intellisense#Citations)

If you prefer to access some of the most common actions through a right click menu, set `latex-workshop.showContextMenu` to `true`. Default is `false`.
