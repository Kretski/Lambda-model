
(base) PS C:\Users\Lenovo\Desktop\Grav\Gr\repo\Lambda-model> git add -f paper3/gw/matched_filter/_archive/README.md
warning: in the working copy of 'paper3/gw/matched_filter/_archive/README.md', LF will be replaced by CRLF the next time Git touches it
(base) PS C:\Users\Lenovo\Desktop\Grav\Gr\repo\Lambda-model> git commit -m "Add archive README: old BEC Level A/B interpretation withdrawn"
[main e56559e] Add archive README: old BEC Level A/B interpretation withdrawn
 1 file changed, 9 insertions(+)
 create mode 100644 paper3/gw/matched_filter/_archive/README.md
(base) PS C:\Users\Lenovo\Desktop\Grav\Gr\repo\Lambda-model> git push
Enumerating objects: 12, done.
Counting objects: 100% (12/12), done.
Delta compression using up to 16 threads
Compressing objects: 100% (7/7), done.
Writing objects: 100% (7/7), 944 bytes | 314.00 KiB/s, done.
Total 7 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (3/3), completed with 3 local objects.
To https://github.com/Kretski/Lambda-model.git
   e3b35a4..e56559e  main -> main
(base) PS C:\Users\Lenovo\Desktop\Grav\Gr\repo\Lambda-model> Get-ChildItem -Filter "-files*" | Remove-Item
(base) PS C:\Users\Lenovo\Desktop\Grav\Gr\repo\Lambda-model> git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
(base) PS C:\Users\Lenovo\Desktop\Grav\Gr\repo\Lambda-model>
