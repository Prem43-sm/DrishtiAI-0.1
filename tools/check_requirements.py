import pkg_resources

def check_requirements(file_path):
    try:
        with open(file_path, "r") as f:
            requirements = [line.strip() for line in f if line.strip()]

        missing = []
        wrong_version = []

        for req in requirements:
            try:
                pkg_resources.require(req)
            except pkg_resources.DistributionNotFound:
                missing.append(req)
            except pkg_resources.VersionConflict as e:
                wrong_version.append(str(e))

        if not missing and not wrong_version:
            print("✅ All requirements are installed with correct versions.")
        else:
            print("\n❌ Requirements issue detected\n")

            if missing:
                print("📦 Missing Packages:")
                for m in missing:
                    print("-", m)

            if wrong_version:
                print("\n⚠ Version Conflicts:")
                for w in wrong_version:
                    print("-", w)

    except FileNotFoundError:
        print("❌ Requirements file not found")


path = input("Enter req.txt file path: ")
check_requirements(path)