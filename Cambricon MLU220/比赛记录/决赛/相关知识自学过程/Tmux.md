

# Tmux 使用教程

### 1.1 会话与进程

命令行的典型使用方式是，打开一个终端窗口（terminal window，以下简称"窗口"），在里面输入命令。**用户与计算机的这种临时的交互，称为一次"会话"（session）** 。

会话的一个重要特点是，窗口与其中启动的进程是[连在一起](https://www.ruanyifeng.com/blog/2016/02/linux-daemon.html)的。打开窗口，会话开始；关闭窗口，会话结束，会话内部的进程也会随之终止，不管有没有运行完。

一个典型的例子就是，[SSH 登录](https://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html)远程计算机，打开一个远程窗口执行命令。这时，网络突然断线，再次登录的时候，是找不回上一次执行的命令的。因为上一次 SSH 会话已经终止了，里面的进程也随之消失了。

为了解决这个问题，会话与窗口可以"解绑"：窗口关闭时，会话并不终止，而是继续运行，等到以后需要的时候，再让会话"绑定"其他窗口。

### 1.2 Tmux 的作用

**Tmux 就是会话与窗口的"解绑"工具，将它们彻底分离。**

> （1）它允许在单个窗口中，同时访问多个会话。这对于同时运行多个命令行程序很有用。
>
> （2） 它可以让新窗口"接入"已经存在的会话。
>
> （3）它允许每个会话有多个连接窗口，因此可以多人实时共享会话。
>
> （4）它还支持窗口任意的垂直和水平拆分。

## 二、基本用法

### 2.1 安装

Tmux 一般需要自己安装。

```bash
# Ubuntu 或 Debian
$ sudo apt-get install tmux
```

### 3.1 新建会话

第一个启动的 Tmux 窗口，编号是`0`，第二个窗口的编号是`1`，以此类推。这些窗口对应的会话，就是 0 号会话、1 号会话。

使用编号区分会话，不太直观，更好的方法是为会话起名。

> ```bash
> $ tmux new -s <session-name>
> ```

上面命令新建一个指定名称的会话。

### 3.2 分离会话

在 Tmux 窗口中，按下`Ctrl+b d`或者输入`tmux detach`命令，就会将当前会话与窗口分离。

> ```bash
> $ tmux detach
> ```

上面命令执行后，就会退出当前 Tmux 窗口，但是会话和里面的进程仍然在后台运行。

`tmux ls`命令可以查看当前所有的 Tmux 会话。

> ```bash
> $ tmux ls
> # or
> $ tmux list-session
> ```

### 3.3 接入会话

`tmux attach`命令用于重新接入某个已存在的会话。

> ```bash
> # 使用会话编号
> $ tmux attach -t 0
> 
> # 使用会话名称
> ```

### 3.4 杀死会话

`tmux kill-session`命令用于杀死某个会话。

> ```bash
> # 使用会话编号
> $ tmux kill-session -t 0
> 
> # 使用会话名称
> $ tmux kill-session -t <session-name>
> ```

## 四、最简操作流程

综上所述，以下是 Tmux 的最简操作流程。

> 1. 新建会话`tmux new -s my_session`。
>
> 2. 在 Tmux 窗口运行所需的程序。
>
> 3. 按下快捷键`Ctrl+b d`将会话分离。
>
>    ```shell
>     tmux detach
>    ```
>
> 4. 下次使用时，重新连接到会话`tmux attach-session -t my_session`。
>
> 5. `tmux split-window`命令用来划分窗格。
>
>    ```
>    # 划分左右两个窗格
>    $ tmux split-window -h
>    ```
>
> 

## 五、窗格操作

Tmux 可以将窗口分成多个窗格（pane），每个窗格运行不同的命令。以下命令都是在 Tmux 窗口中执行。

### 5.1 划分窗格

`tmux split-window`命令用来划分窗格。

> ```bash
> # 划分上下两个窗格
> $ tmux split-window
> 
> # 划分左右两个窗格
> $ tmux split-window -h
> ```

### 5.4 窗格快捷键

下面是一些窗格操作的快捷键。

> - `Ctrl+b %`：划分左右两个窗格。
> - `Ctrl+b "`：划分上下两个窗格。
> - `Ctrl+b <arrow key>`：光标切换到其他窗格。`<arrow key>`是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键`↓`。
> - `Ctrl+b ;`：光标切换到上一个窗格。
> - `Ctrl+b o`：光标切换到下一个窗格。
> - `Ctrl+b {`：当前窗格与上一个窗格交换位置。
> - `Ctrl+b }`：当前窗格与下一个窗格交换位置。
> - `Ctrl+b Ctrl+o`：所有窗格向前移动一个位置，第一个窗格变成最后一个窗格。
> - `Ctrl+b Alt+o`：所有窗格向后移动一个位置，最后一个窗格变成第一个窗格。
> - `Ctrl+b x`：关闭当前窗格。
> - `Ctrl+b !`：将当前窗格拆分为一个独立窗口。
> - `Ctrl+b z`：当前窗格全屏显示，再使用一次会变回原来大小。
> - `Ctrl+b Ctrl+<arrow key>`：按箭头方向调整窗格大小。
> - `Ctrl+b q`：显示窗格编号。

## 六、窗口管理

除了将一个窗口划分成多个窗格，Tmux 也允许新建多个窗口。

### 6.1 新建窗口

`tmux new-window`命令用来创建新窗口。

> ```bash
> $ tmux new-window
> 
> # 新建一个指定名称的窗口
> $ tmux new-window -n <window-name>
> ```

### 6.2 切换窗口

`tmux select-window`命令用来切换窗口。

> ```bash
> # 切换到指定编号的窗口
> $ tmux select-window -t <window-number>
> 
> # 切换到指定名称的窗口
> $ tmux select-window -t <window-name>
> ```

## Ubuntu Tmux 启用鼠标滚动

在Ubuntu上使用Tmux是一件非常舒服的事，但有时使用鼠标滚轮时，和平时使用终端的习惯不怎么一致，因此可以设置启用鼠标滚轮。
具体方式：
按完前缀ctrl+B后，再按冒号：进入命令行模式，
输入以下命令：

set -g mouse on

就启用了鼠标滚轮，可以通过鼠标直接选择不同的窗口，也可以上下直接翻页。

Tip
但在以上设置下，会发现无法用中键向 tmux 中复制文本，也无法将 tmux 中选择好的文本中键复制到系统其他应用程序中。
这里有一个 trick，那就是在 tmux 中不论选择还是复制时，都按住 Shift 键，你会发现熟悉的中键又回来了 ? 此外，还可以使用 Shift+Insert 快捷键将系统剪切板中的内容输入 tmux 中。 相对于 tmux 原生的选择模式（不加 shift 键），使用系统选择有个缺陷，即当一行内存在多个面板时，无法选择单个面板

