



### 分布式版本控制系统

Git 是一种分布式版本控制系统 (Distributed Version Control System DVCS) 。这种系统下，客户端不只是简单地拉取某个版本的文件，而是把整个记录文件版本的数据库（即整个代码仓库）都克隆到本地系统上。这样以来，任何一处协同工作用的服务器发生故障，事后都可以用任何一个镜像出来的本地仓库恢复。因为每一次的克隆工作，实际上都是一次对代码仓库的完整备份。



### 使用GIT进行管理

#### 创建本地仓库

git init

\1) 配置用户身份

在Git Bash中，输入如下指令 ：

![img](https://img-blog.csdn.net/20170305003843187?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

此操作在多人协作时非常有用，可以用来标识更新代码的用户的身份。

\2) 提交代码到本地仓库
使用“git add”命令来添加要提交的文件。 

语法：git add .（表示添加所有文件）|目录名|文件名

![img](https://img-blog.csdn.net/20170305004153395?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



添加文件后，输入指令“git status” 来查看当前仓库中的文件状态。 

使用“git commit”命令来提交文件。 

语法：git commit -m “提交描述信息” 



#### 查看文件更新状态

1)“git status”命令用来查看本地文件和当前版本的文件有哪些不同 。

当有新文件添加进来时：

![img](https://img-blog.csdn.net/20170305004507696?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

当有文件被修改时：

![img](https://img-blog.csdn.net/20170305004532680?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

当有文件被删除时：

![img](https://img-blog.csdn.net/20170305004556588?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

\2) “git diff”命令用来查看文件发生修改的具体内容，减号表示减少的部分，加号表示增加的部分。 



#### 撤销操作

1)文件修改的撤销
使用“git checkout”命令可以将发生修改的文件恢复到当前版本未修改时的状态，相当于svn中的“revert”操作。

语法：git checkout -- file

![img](https://img-blog.csdn.net/20170305004644633?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

2)新增文件的撤销

使用“git reset HEAD”命令可以撤销未提交的“git add”操作。

语法：git reset HEAD file

![img](https://img-blog.csdn.net/20170305004712368?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

#### 日志查看

使用“git log”命令查看提交记录日志。

![img](https://img-blog.csdn.net/20170305004738509?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

使用“git log id -p”命令查看当次提交具体的修改内容。

![img](https://img-blog.csdn.net/20170305004801982?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGVuZ2p1bmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

按“q”键退出日志查看。



参考:

https://bbs.csdn.net/skill/gml/gml-f0d68868583b48d0851c7add33e2f8ff

https://blog.csdn.net/Jack_CJZ/article/details/80934750

https://blog.csdn.net/pengjunlee/article/details/60376961



### 使用GIT上传至自己的GitHub

https://blog.csdn.net/yaxin3690/article/details/53840953





### 使用Github Desktop管理代码

https://www.jianshu.com/p/6063974849db



### 关于branch的理解

GitHub仓库默认有一个master的分支，当我们在master分支开发过程中接到一个新的功能需求，我们就可以新建一个分支同步开发而互不影响，开发完成后，在合并merge到主分支master上。

https://www.cnblogs.com/yanliujun-tangxianjun/p/5740704.html



