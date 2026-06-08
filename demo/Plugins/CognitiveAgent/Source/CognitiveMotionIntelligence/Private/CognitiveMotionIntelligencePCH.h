// PCH privado do módulo CognitiveMotionIntelligence.
// Mantém o módulo isolado do PCH compartilhado da engine, impedindo que os
// headers da LibTorch vazem para outros módulos (como o módulo de jogo CMI).
#pragma once

// CAUSA RAIZ do erro 'gflags/gflags.h not found':
// O c10/util/Flags.h faz `#ifdef C10_USE_GFLAGS` → `#include <gflags/gflags.h>`.
// O teste é #ifdef (existência), NÃO #if (valor). Definir C10_USE_GFLAGS=0
// AINDA satisfaz o #ifdef e dispara o include. A forma correta de desligar é
// garantir que estas macros NÃO existam — por isso usamos #undef aqui, como
// rede de segurança caso algum módulo as defina por engano.
#ifdef C10_USE_GFLAGS
  #undef C10_USE_GFLAGS
#endif
#ifdef C10_USE_GLOG
  #undef C10_USE_GLOG
#endif

#include "CoreMinimal.h"
