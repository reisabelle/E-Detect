package com.example.thesis

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class diseases : AppCompatActivity(){
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.diseases_info)

        val info = findViewById<TextView>(R.id.diseaseTxt)

        info.text = "To support you further, E-Detect provides brief descriptions of each detected disease. Here’s a quick overview of what E-Detect can help identify:\n" +
                "\nBUMBLEFOOT" +
                "\nA bacterial infection commonly affecting the foot, Bumblefoot can cause swelling, redness, and lameness. It is often caused by a bacterial infection entering through cuts or abrasions. Early detection can prevent further complications, as untreated Bumblefoot can severely impact a chicken’s mobility." +
                "\nDistinguishable features: Slight swelling, small scab || Larger, darker scab, limping || and Severe swelling, and abscesses\n" +

                "\nFOWLPOX" +
                "\nA viral disease that appears as wart-like lesions, primarily on the comb, wattles, and around the eyes. Fowlpox spreads quickly through a flock, especially in hot, humid conditions, making early identification crucial to control its spread." +
                "\nDistinguishable features: Small wart-like lesions (dry form) || Larger, scabbing lesions\n" +


                "\nCORYZA" +
                "\nA respiratory illness causing swelling, nasal discharge, and decreased egg production. Coryza spreads through contaminated feed and water, so quick detection can limit transmission and aid in prompt isolation and treatment." +
                "\nDistinguishable features: face swelling || pronounced face swelling || Severe swelling\n" +

                "\nIMAGE QUALITY & TIPS FOR BEST RESULTS" +
                "\nFor optimal accuracy, consider the following tips:\n" +
                "\n(1) Capture Clear, Well-Lit Images: Natural lighting and clear angles help the app analyze the image more effectively." +
                "\n(2) Focus on Affected Areas: If possible, capture images of areas where symptoms appear (like the feet for Bumblefoot or the face for Fowlpox and Coryza)." +
                "\n(3) Avoid Background Distractions: An uncluttered background helps the model focus on the chicken itself, improving accuracy."
        }
}