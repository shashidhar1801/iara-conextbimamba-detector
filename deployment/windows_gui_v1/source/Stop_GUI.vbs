Option Explicit
Dim shell, fso, root, scriptPath, command, exitCode
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = root & "\Stop_GUI.ps1"
shell.CurrentDirectory = root
command = "powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " & Chr(34) & scriptPath & Chr(34)
exitCode = shell.Run(command, 0, True)

If exitCode = 0 Then
    MsgBox "The ConExtBiMamba GUI server is stopped.", vbInformation, "ConExtBiMamba GUI"
Else
    MsgBox "The server could not be stopped safely. Check runs\gui_server_state.json and logs\gui_launcher.log.", vbCritical, "ConExtBiMamba GUI"
End If
WScript.Quit exitCode

