#### LaTex中的一些长宽
`\hsize`: 是 `\TeX` 中定义的长度，是一种叫做水平盒子的长度，它的主要作用是告诉TeX系统什么时候换行。所以大部分时候和`\textwidth`是一致的，但是在分栏状况下，`\hsize`只是栏的宽度；
`\textwidth`:是 `\LaTeX ` 中定义的长度，等效于`\hsize`，并且是固定不变的，可以理解为一行文字的宽度。
`\pagewidth`:包含了页边的宽度，比\textwidth要大。
`\linewidth`:这指得是目前环境的宽度，是依赖于上下文的一个宽度值，例如新建了一个box，在这个box中，\linewidth是box中文字的宽度。再例如minipage环境中，\linewidth就和这个minipage的大小有关.
`\columnwidth`: 如果文章分栏的话，这个宽度就是每一栏的宽度。