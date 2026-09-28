# 用 SSH 将本项目上传到 GitHub

适用于当前 AutoDL / Linux / Bash 环境。编写此文档时，项目目录尚未初始化为 Git 仓库。以下步骤由你在终端执行；本文档不会创建 GitHub 仓库、生成密钥或自动推送。

## 1. 准备 SSH 密钥

先检查现有密钥，已有可用密钥可直接复用：

```bash
ls -la ~/.ssh
```

如果目录不存在，或没有要复用的密钥，运行下列命令。将邮箱替换为自己的邮箱（用作密钥注释）：

```bash
ssh-keygen -t ed25519 -C "YOUR_EMAIL@example.com"
```

按提示选择保存位置和口令。默认文件名为 `~/.ssh/id_ed25519`；如果提示覆盖已有文件，请拒绝并选择其他文件名。以下命令假定使用默认路径；自定义路径需同步替换。

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

复制 `.pub` 文件的完整一行。在 GitHub 打开 **Settings → SSH and GPG keys → New SSH key**，选择 **Authentication Key**，粘贴并保存。上传的是公钥；不分享或提交没有 `.pub` 后缀的私钥。

这些步骤依据 GitHub 官方的[密钥生成说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux)与[添加公钥说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)。

## 2. 测试连接

```bash
ssh -T git@github.com
```

首次连接时，按 GitHub 官方[连接测试说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)核对显示的主机指纹再接受。显示 `Hi USERNAME! You've successfully authenticated...` 表示认证成功；GitHub 不提供交互式 shell，成功测试仍可能返回状态码 1。

如果服务器网络屏蔽了 22 端口，可以测试官方提供的 443 端口：

```bash
ssh -T -p 443 git@ssh.github.com
```

成功后，后面的远程地址可以直接改用：

```text
ssh://git@ssh.github.com:443/YOUR_USERNAME/self-evolve.git
```

详见 [GitHub SSH over HTTPS port](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port)。

## 3. 在 GitHub 建立空仓库

登录 GitHub，点击 **New repository**，填写仓库名，例如 `self-evolve`，选择 Public 或 Private。

为了直接推送本地已有项目，这一步不要勾选自动添加 README、`.gitignore` 或 License。创建后复制 SSH 地址，例如：

```text
git@github.com:YOUR_USERNAME/self-evolve.git
```

本地已有 README 和 `.gitignore`。项目当前没有许可证文件；如果希望明确允许他人使用、修改、分发代码，应自行选择许可证并添加 `LICENSE`。此步骤不替你决定授权范围。

## 4. 初始化并检查待上传内容

```bash
cd /root/autodl-tmp/self-evolve
git init -b main
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL@example.com"
git add .
git status --short
git diff --cached --stat
```

这里的 `git config` 仅设置本仓库的提交身份。邮箱也可以使用 GitHub 邮箱设置中提供的 noreply 地址。

`.gitignore` 排除了虚拟环境、`data/`、`models/`、`runs/`、环境变量文件、日志及备份。用于展示的实验摘要保存在 `docs/results/`，会正常上传。因此仓库包含代码、配置、文档和小型结果快照；读者可根据 README 重新准备数据和模型。

首次提交前核对待上传文件，尤其是自己新增过的配置和脚本是否包含 API key。不要使用 `git add -f` 强制上传整个 `runs/` 或 `models/`。如果误暂存了某文件，可在保留工作区文件的前提下移出暂存区：

```bash
git rm --cached -- PATH_TO_FILE
```

确认后提交：

```bash
git commit -m "Add Jev-guided subspace adaptation experiments"
```

## 5. 添加远程仓库并推送

替换 `YOUR_USERNAME` 和仓库名称：

```bash
git remote add origin git@github.com:YOUR_USERNAME/self-evolve.git
git remote -v
git push -u origin main
```

如果使用上述 443 端口通道，`git remote add origin` 后面改为对应的 `ssh://git@ssh.github.com:443/...` 地址。

上传后刷新 GitHub 页面，根目录 `README.md` 会自动展示。工作流参考 GitHub 官方[上传本地代码说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 6. 后续更新和常见问题

以后修改完成后执行：

```bash
git add .
git diff --cached --stat
git commit -m "Describe your changes"
git push
```

- `Permission denied (publickey)`：检查公钥是否添加到正确 GitHub 账号、`ssh-add -l` 是否列出对应密钥，以及 SSH 使用的私钥路径是否正确。
- `Repository not found`：核对用户名、仓库名、仓库是否已创建，以及当前 SSH 账号是否有写入权限。
- `remote origin already exists`：先用 `git remote -v` 检查；需要修改时运行 `git remote set-url origin NEW_SSH_URL`。
- 推送报 `non-fast-forward`：远程可能已经有 README 或其他提交。先 `git fetch origin`，用 `git log --oneline --graph --all` 检查历史，再决定如何合并。不要用强制推送覆盖不明的远程历史。
- 私钥或 API key 已经提交：仅添加 `.gitignore` 不会清除历史中的内容；先更换泄露凭据，再处理 Git 历史。
