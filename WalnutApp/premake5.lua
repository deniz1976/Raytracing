project "WalnutApp"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   targetdir "bin/%{cfg.buildcfg}"
   staticruntime "off"

   files { "src/**.h", "src/**.cpp", "src/**.comp" }

   includedirs
   {
      "src",
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

   -- The shader is compiled twice: with ray queries for devices that support
   -- them, and without as the fallback the renderer picks everywhere else.
   local VulkanSDK = os.getenv("VULKAN_SDK")
   local glslc = "\"" .. VulkanSDK .. "/Bin/glslc.exe\" --target-env=vulkan1.2"
   local shaderSource = "\"%{wks.location}/WalnutApp/src/Shaders/RayTracing.comp\""
   local shaderOutput = "%{wks.location}/WalnutApp/assets/shaders/"
   prebuildcommands
   {
      "{MKDIR} \"%{wks.location}/WalnutApp/assets/shaders\"",
      glslc .. " -DUSE_RAY_QUERY " .. shaderSource .. " -o \"" .. shaderOutput .. "RayTracing.comp.spv\"",
      glslc .. " " .. shaderSource .. " -o \"" .. shaderOutput .. "RayTracingFallback.comp.spv\""
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
