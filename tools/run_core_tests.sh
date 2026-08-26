#!/usr/bin/env bash
#
# CielScore — execution des tests de conformite du moteur de calcul SANS SDK Android.
#
# Le coeur de l'application (ephemerides, crepuscules, formules, moteur de score)
# est du Kotlin pur : il se compile et se teste avec le seul compilateur Kotlin,
# telecharge depuis Maven Central. Utile pour verifier la conformite a la
# documentation sur une machine ou Android Studio n'est pas installe.
#
# Usage :  ./tools/run_core_tests.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/android/app/src/main/java/com/cielscore/app"
TEST="$ROOT/android/app/src/test/java/com/cielscore/app"
WORK="${CIELSCORE_WORK:-$ROOT/.cielscore-core-tests}"
LIB="$WORK/lib"

KOTLIN_VERSION=2.0.21
COROUTINES_VERSION=1.8.1
MAVEN=https://repo1.maven.org/maven2

mkdir -p "$LIB"

fetch() { # url, destination
  if [ ! -f "$2" ]; then
    echo "  telechargement de $(basename "$2")"
    curl -fsSL --retry 3 -o "$2" "$1"
  fi
}

echo "Preparation des dependances"
fetch "$MAVEN/org/jetbrains/kotlin/kotlin-compiler/$KOTLIN_VERSION/kotlin-compiler-$KOTLIN_VERSION.jar" "$LIB/kotlin-compiler.jar"
fetch "$MAVEN/org/jetbrains/kotlin/kotlin-stdlib/$KOTLIN_VERSION/kotlin-stdlib-$KOTLIN_VERSION.jar" "$LIB/kotlin-stdlib.jar"
fetch "$MAVEN/org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/$COROUTINES_VERSION/kotlinx-coroutines-core-jvm-$COROUTINES_VERSION.jar" "$LIB/coroutines.jar"
fetch "$MAVEN/org/jetbrains/intellij/deps/trove4j/1.0.20200330/trove4j-1.0.20200330.jar" "$LIB/trove4j.jar"
fetch "$MAVEN/org/jetbrains/annotations/23.0.0/annotations-23.0.0.jar" "$LIB/annotations.jar"
fetch "$MAVEN/junit/junit/4.13.2/junit-4.13.2.jar" "$LIB/junit.jar"
fetch "$MAVEN/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar" "$LIB/hamcrest.jar"

COMPILER_CP="$LIB/kotlin-compiler.jar:$LIB/coroutines.jar:$LIB/trove4j.jar:$LIB/annotations.jar"
RUNTIME_CP="$LIB/kotlin-stdlib.jar:$LIB/coroutines.jar:$LIB/junit.jar:$LIB/hamcrest.jar"

kotlinc() {
  java -cp "$COMPILER_CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler \
    -no-stdlib -no-reflect -nowarn "$@"
}

echo "Compilation du moteur de calcul"
rm -rf "$WORK/classes" "$WORK/test-classes"
kotlinc -cp "$RUNTIME_CP" -d "$WORK/classes" \
  "$SRC/astro/AstroMath.kt" \
  "$SRC/astro/SolarSystem.kt" \
  "$SRC/astro/Twilight.kt" \
  "$SRC/astro/SkyProjection.kt" \
  "$SRC/astro/SkyCalendar.kt" \
  "$SRC/scoring/Formulas.kt" \
  "$SRC/scoring/ScoringEngine.kt" \
  "$SRC/catalog/SkyObject.kt" \
  "$SRC/model/SessionModels.kt" \
  "$SRC/model/SmartTelescope.kt"

echo "Compilation des tests"
kotlinc -cp "$RUNTIME_CP:$WORK/classes" -d "$WORK/test-classes" \
  "$TEST/DocumentationConformanceTest.kt"

echo "Execution"
java -cp "$RUNTIME_CP:$WORK/classes:$WORK/test-classes" \
  org.junit.runner.JUnitCore com.cielscore.app.DocumentationConformanceTest
