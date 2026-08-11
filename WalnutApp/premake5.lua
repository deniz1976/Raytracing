project "WalnutApp"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   targetdir "bin/%{cfg.buildcfg}"
   staticruntime "off"

   files { "src/**.h", "src/**.cpp", "src/**.comp" }

   includedirs
   {
      "../vendor/imgui",
      "../vendor/glfw/include",

      "../Walnut/src",

      "%{IncludeDir.VulkanSDK}",
      "%{IncludeDir.glm}",
   }

    links
    {
        "Walnut"
    }

   targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
   objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")
   debugdir "%{wks.location}/WalnutApp"

   local VulkanSDK = os.getenv("VULKAN_SDK")
   prebuildcommands
   {
      "{MKDIR} \"%{wks.location}/WalnutApp/assets/shaders\"",
      "\"" .. VulkanSDK .. "/Bin/glslc.exe\" --target-env=vulkan1.2 \"%{wks.location}/WalnutApp/src/Shaders/RayTracing.comp\" -o \"%{wks.location}/WalnutApp/assets/shaders/RayTracing.comp.spv\""
   }

   filter "system:windows"
      systemversion "latest"
      defines { "WL_PLATFORM_WINDOWS" }

   filter "configurations:Debug"
      defines { "WL_DEBUG" }
      runtime "Debug"
      symbols "On"

   filter "configurations:Release"
      defines { "WL_RELEASE" }
      runtime "Release"
      optimize "On"
      symbols "On"

   filter "configurations:Dist"
      kind "WindowedApp"
      defines { "WL_DIST" }
      runtime "Release"
      optimize "On"
      symbols "Off"
