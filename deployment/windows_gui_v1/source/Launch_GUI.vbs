Option Explicit
Dim shell, fso, root, pythonw, launcher, installer, command, answer, exitCode
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.gui-venv\Scripts\pythonw.exe"
launcher = root & "\launcher.py"
installer = root & "\Install_GUI.vbs"

If Not fso.FileExists(pythonw) Then
    answer = MsgBox("The private GUI environment is not installed. Install it now? Installation runs without a console window and can take several minutes.", vbQuestion + vbYesNo, "ConExtBiMamba GUI")
    If answer <> vbYes Then
        WScript.Quit 2
    End If
    If Not fso.FileExists(installer) Then
        MsgBox "Install_GUI.vbs is missing. Extract the complete deployment ZIP again.", vbCritical, "ConExtBiMamba GUI"
        WScript.Quit 3
    End If
    command = "wscript.exe " & Chr(34) & installer & Chr(34)
    exitCode = shell.Run(command, 0, True)
    If exitCode <> 0 Or Not fso.FileExists(pythonw) Then
        MsgBox "Installation did not complete, so the GUI was not started. Review logs\install.log.", vbCritical, "ConExtBiMamba GUI"
        WScript.Quit 4
    End If
End If

shell.CurrentDirectory = root
command = Chr(34) & pythonw & Chr(34) & " " & Chr(34) & launcher & Chr(34)
shell.Run command, 0, False
