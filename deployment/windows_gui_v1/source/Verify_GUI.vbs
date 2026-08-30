Option Explicit
Dim shell, fso, root, pythonw, verifier, command, exitCode
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.gui-venv\Scripts\pythonw.exe"
verifier = root & "\scripts\verify_deployment.py"

If Not fso.FileExists(pythonw) Then
    MsgBox "The private GUI environment is not installed. Run Install_GUI.vbs first.", vbExclamation, "ConExtBiMamba GUI"
    WScript.Quit 2
End If

shell.CurrentDirectory = root
command = Chr(34) & pythonw & Chr(34) & " " & Chr(34) & verifier & Chr(34) & " --quick"
exitCode = shell.Run(command, 0, True)

If exitCode = 0 Then
    MsgBox "Package integrity and frozen inference contract verification passed. See logs\verification_latest.json.", vbInformation, "ConExtBiMamba GUI"
Else
    MsgBox "Verification failed. See logs\verification_latest.json for details.", vbCritical, "ConExtBiMamba GUI"
End If
WScript.Quit exitCode

