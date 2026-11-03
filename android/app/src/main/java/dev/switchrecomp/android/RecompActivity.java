package dev.switchrecomp.android;

import android.content.pm.ActivityInfo;
import android.os.Build;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;

import org.libsdl.app.SDLActivity;

/**
 * SDL3 activity for the recomp host.
 *
 * <p>SDLActivity loads {@code libSDL3.so} and then {@code libmain.so} (our Rust
 * bootstrap), and resolves {@code SDL_main} in that library. The bootstrap
 * loads {@code librecomp_game.so} itself, so the generated title library is
 * not listed here: replacing the title does not require changing Java.
 *
 * <p>The title is a full-screen landscape game: the system bars are hidden
 * (otherwise they take 110 px off one edge and 132 px off the other, and the
 * 16:9 frame is shrunk into what is left), the display cutout is used, and
 * both landscape directions follow the sensor so turning the phone over
 * flips the picture instead of leaving it upside down.
 */
public class RecompActivity extends SDLActivity {
    @Override
    protected String[] getLibraries() {
        return new String[] { "SDL3", "main" };
    }

    @Override
    protected String getMainSharedObject() {
        return getContext().getApplicationInfo().nativeLibraryDir + "/libmain.so";
    }

    @Override
    protected String getMainFunction() {
        return "SDL_main";
    }

    @Override
    protected void onResume() {
        super.onResume();
        useFullScreen();
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            useFullScreen();
        }
    }

    /**
     * SDL asks for the orientation its hint allows. With both landscape
     * directions in {@code SDL_ORIENTATIONS}, prefer the sensor over the
     * user's rotation lock ({@code SCREEN_ORIENTATION_USER_LANDSCAPE}): a
     * locked activity never turns over, which is what leaves the picture
     * upside down when the phone is held the other way.
     */
    @Override
    public void setOrientationBis(int w, int h, boolean resizable, String hint) {
        super.setOrientationBis(w, h, resizable, hint);
        if (hint != null
                && hint.contains("LandscapeLeft")
                && hint.contains("LandscapeRight")) {
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE);
        }
    }

    private void useFullScreen() {
        Window window = getWindow();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            window.setAttributes(attributes);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Lay the content out over the whole display, not inside the
            // bars' insets: hiding the bars alone leaves the SDL surface at
            // 2158x970 instead of 2400x1080.
            window.setDecorFitsSystemWindows(false);
            WindowInsetsController controller = window.getInsetsController();
            if (controller != null) {
                controller.hide(
                        WindowInsets.Type.statusBars() | WindowInsets.Type.navigationBars());
                controller.setSystemBarsBehavior(
                        WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            }
            return;
        }
        window.getDecorView()
                .setSystemUiVisibility(
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                                | View.SYSTEM_UI_FLAG_FULLSCREEN
                                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }
}
