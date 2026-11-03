plugins {
    id("com.android.application")
}

android {
    namespace = "dev.switchrecomp.android"
    compileSdk = 35
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "dev.switchrecomp.android"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        debug {
            isDebuggable = true
        }
        release {
            isMinifyEnabled = false
            // Sign a release APK with the project's debug key only when the
            // caller supplies one; the direct scripts/build-apk.sh path does
            // this explicitly with apksigner.
        }
    }

    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")
    sourceSets["main"].assets.srcDirs("src/main/assets")

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    packaging {
        // extractNativeLibs=true in the manifest needs the legacy packaging
        // layout; it also keeps plain adb pushes of libSDL3.so predictable.
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

// Build the Rust bootstrap and SDL3 before Gradle collects jniLibs. The script
// also extracts the official SDL3 Java sources from the SDL3 build into
// src/main/java/org/libsdl/app (generated, not committed).
tasks.register<Exec>("buildNative") {
    workingDir = file("../")
    commandLine("scripts/build-native.sh", "--release")
}

tasks.named("preBuild") {
    dependsOn("buildNative")
}
