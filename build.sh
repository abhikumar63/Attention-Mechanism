set -e

echo "Cleaning previous builds..."
rm -rf build
mkdir build
cd build

echo "Generating Makefiles..."
cmake ..

echo "Compiling the engine..."
make

echo -e "\nExecuting binary...\n"
./attention_engine