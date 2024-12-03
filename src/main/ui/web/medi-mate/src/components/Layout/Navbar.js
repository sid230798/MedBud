import React from "react";

const Navbar = () => {
    return (
        <div className="flex h-[50px] sm:h-[70px] border-t border-transparent py-2 px-8 items-center bg-gradient-to-r from-purple-300 via-pink-300 to-purple-300 bg-clip-border">
            {/* Lottie Animation */}
            <iframe
                src="https://lottie.host/embed/b4700158-2f71-4010-bba0-9ce292ea042a/YUj9CEyXya.json"
                width="40"
                height="40"
                allowFullScreen
                className="mr-4"
            ></iframe>

            {/* Navbar Text */}
            <div className="font-bold text-xl sm:text-2xl text-white">
                <a className="hover:opacity-50" href="#">
                    MedBud
                </a>
            </div>
        </div>
    );
};

export default Navbar;
