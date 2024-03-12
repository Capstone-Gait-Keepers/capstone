<template>
<div class="logo">
  <img alt="WalkWise Logo" class="logo" src="@/assets/logo.svg" width="125" height="125" />
  <p>WalkWise</p>
</div>
</template>

<style scoped>
.logo {
  display: flex;
  align-items: center;
  height: 2rem;
  width: auto;
}

@media (min-width: 1024px) {
  .logo {
    margin-right: .5rem;
  }
}
</style>