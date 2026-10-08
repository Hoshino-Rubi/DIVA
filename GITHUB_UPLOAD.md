# 将此源码目录上传到 GitHub

此目录是从工作区准备的独立源码副本。不要在原始研究目录执行 git add。
已保留代码、测试、桌面项目、协议和说明，排除模型、虚拟环境、运行缓存、
个人音频、人工工程、聊天交接记录，以及包含个人源音频路径的历史结果 JSON。
少量测试/文档中的用户名仅在此副本替换为 Example User。

## 新建仓库并上传

Git 和 GitHub CLI 已安装。本机源码目录已初始化 main 分支；ZIP 不含 .git，解压后按下面命令初始化。没有设置远端或执行上传。
本机没有 Git 作者姓名和邮箱配置，先在当前仓库设置（不改全局设置）。

在 Windows 命令提示符（cmd）中，进入解压后的源码根目录，再执行：

~~~bat
git init -b main
git config user.name "YOUR_GITHUB_NAME"
git config user.email "YOUR_COMMIT_EMAIL"
git add .
git commit -m "Initial source release"
gh auth login --hostname github.com --git-protocol https --web
gh repo create any2xsampa --private --source=. --remote=origin --push
~~~

将姓名和邮箱替换为你的提交身份。若明确要公开，把最后一行 --private 改成 --public。
仓库名称也可以改成你希望的名称。登录通过 GitHub CLI 的官方浏览器流程完成，
不要把密码或访问令牌粘贴进源码或聊天。

已有仓库时，跳过 gh repo create，改用自己的仓库地址：

~~~bat
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
~~~

上述已有仓库流程适用于空仓库。非空仓库不要使用 force push；先处理已有历史。
若 GitHub 网页新建仓库供此流程使用，不要额外创建 README 或 .gitignore，本地已经包含。

如你本机需要使用已有代理，在当前 cmd 窗口设置即可：

~~~bat
set HTTPS_PROXY=http://127.0.0.1:18080
set HTTP_PROXY=http://127.0.0.1:18080
~~~

这些只是操作说明；准备源码包时没有登录、创建远端仓库、改账号权限或推送任何东西。

## 发布内容

第一次建议发布源码。代码可以浏览/下载，但不会附带本机 Python、模型权重或训练数据。
README 中的模型路径需由使用者配置；历史研究工具需要各自缓存，本包不包含这些缓存。
现有开发机桌面目录依赖原工作区，不能当作已验证的跨机器独立安装器发布。

根目录 requirements.lock.txt 记录历史完整开发环境，并非最小安装依赖。
其中 editable 路径已经改为 -e .，其余版本未更改。根据实际使用的适配器安装依赖。
当前 normal CLI 与桌面共享本地 stdio 引擎。

本包没有替项目选择软件许可证。若你希望作为允许他人复用的开源项目发布，
请决定适合项目的 LICENSE；第三方库和模型仍分别保留其各自许可。

## 验证

本次仅验证源码整理与已有结果导出，不执行新模型推理、训练、下载或新增人工评测。
哈希清单和本机验证记录保存在原工作区的独立 github-source-* run 内，
不把含私人绝对路径的内部清单上传到仓库。
