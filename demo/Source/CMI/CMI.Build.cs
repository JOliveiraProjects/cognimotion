// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class CMI : ModuleRules
{
	public CMI(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
	
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "EnhancedInput" });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });
		
		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
		
		// ── ADICIONE ISTO ──
		PublicDefinitions.Add("C10_USE_GLOG=0");
		PublicDefinitions.Add("C10_USE_GFLAGS=0");
		PublicDefinitions.Add("C10_USE_MINIMAL_GLOG=1");
		PublicDefinitions.Add("GLOG_NO_ABBREVIATED_SEVERITIES=1");
		PublicDefinitions.Add("GOOGLE_GLOG_DLL_DECL=");
		PublicDefinitions.Add("NOMINMAX");
		bUseRTTI = true;
		bEnableExceptions = true;
	}
}
