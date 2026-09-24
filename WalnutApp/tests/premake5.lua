-- Unit tests for the scene modules that need no GPU. Run the executable from any
-- directory; it only writes to the system temporary directory.
project "SceneTests"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   staticruntime "off"

   files
   {
      "SceneTests.cpp",
      "../src/Scene/**.h",
      "../src/Scene/**.cpp",
   }

   includedirs
   {
      "../src",
      "../../vendor/glm",
   }

   targetdir ("../../bin/" .. outputdir .. "/%{prj.name}")
   objdir ("../../bin-int/" .. outputdir .. "/%{prj.name}")

   filter "system:windows"
      systemversion "latest"

   filter "configurations:Debug"
      runtime "Debug"
      symbols "On"

   filter "configurations:Release or configurations:Dist"
      runtime "Release"
      optimize "On"
